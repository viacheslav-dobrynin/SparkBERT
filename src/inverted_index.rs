use std::collections::HashMap;

use anyhow::Result;
use float8::F8E4M3;
use tantivy::{
    directory::RamDirectory,
    query::{BooleanQuery, Query},
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, FAST, STORED,
    },
    Index, IndexReader, ReloadPolicy, Searcher, SingleSegmentIndexWriter, TantivyDocument, Term,
};

use crate::tf_term_query::TfTermQuery;

pub struct InvertedIndex {
    index: Option<Index>,
    writer: Option<SingleSegmentIndexWriter>,
    pub reader: Option<IndexReader>,
    token_cluster_id: Field,
    doc_id: Field,

    pending: HashMap<u64, TantivyDocument>,
}

impl InvertedIndex {
    pub fn open() -> Result<Self> {
        let mut schema_builder = Schema::builder();
        let tok_opts = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::WithFreqs),
        );
        let token_cluster_id = schema_builder.add_text_field("token", tok_opts);
        let doc_id = schema_builder.add_u64_field("doc_id", FAST | STORED);
        let schema = schema_builder.build();

        let writer = Index::builder()
            .schema(schema)
            .single_segment_index_writer(RamDirectory::create(), 500_000_000)?; // 500 MB heap
        let writer = Some(writer);

        Ok(Self {
            index: None,
            writer,
            reader: None,
            token_cluster_id,
            doc_id,
            pending: HashMap::new(),
        })
    }

    pub fn add_pair(&mut self, token_cluster_id: &str, doc_id: u64, score: f32) -> Result<()> {
        let reps = F8E4M3::from_f32(score).to_bits();
        if reps == 0 {
            return Ok(());
        }

        let doc_entry = self.pending.entry(doc_id).or_insert_with(|| {
            let mut d = tantivy::TantivyDocument::new();
            d.add_u64(self.doc_id, doc_id);
            d
        });

        for _ in 0..reps {
            doc_entry.add_text(self.token_cluster_id, token_cluster_id);
        }
        Ok(())
    }

    /// commit
    pub fn finalize(&mut self) -> Result<()> {
        let mut writer = self.writer.take().unwrap();
        for doc in self.pending.values() {
            writer.add_document(doc.clone())?;
        }
        self.pending.clear();
        println!("Inverted index memory usage: {}", &writer.mem_usage());
        let index = writer.finalize()?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        self.index = Some(index);
        self.reader = Some(reader);
        Ok(())
    }

    pub fn get_num_docs(&mut self) -> Result<u64> {
        let reader = self.reader.as_ref().unwrap();
        let searcher = reader.searcher();
        Ok(searcher.num_docs())
    }

    // TODO: 1. построить графики качество/время 2. посмотреть на глубину обхода постинг листов
    /// execute a query that is a list of `(token#cluster)` strings
    /// returns `Vec<(doc_id, sum_score)>` sorted desc by sum_score
    pub fn search(
        &mut self,
        searcher: Option<&Searcher>,
        pairs: &[&str],
        top_k: usize,
    ) -> Result<Vec<(u64, f64)>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let searcher = if let Some(searcher) = searcher {
            searcher
        } else {
            &self.reader.as_ref().unwrap().searcher()
        };
        let mut clauses = Vec::with_capacity(pairs.len());
        for &tok in pairs {
            let term = Term::from_field_text(self.token_cluster_id, tok);
            clauses.push(Box::new(TfTermQuery::new(term)) as Box<dyn Query>);
        }
        let bool_q = BooleanQuery::union(clauses);

        let hits = searcher.search(&bool_q, &tantivy::collector::TopDocs::with_limit(top_k))?;

        let mut results = Vec::with_capacity(hits.len());
        for (score, doc_addr) in hits {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_addr)?;
            let doc_id: u64 = retrieved_doc
                .get_first(self.doc_id)
                .unwrap()
                .as_u64()
                .unwrap();
            results.push((doc_id, score as f64));
        }
        Ok(results)
    }
}
