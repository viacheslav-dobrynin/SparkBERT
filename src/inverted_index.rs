use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

use anyhow::{Context, Result};
use float8::F8E4M3;
use tantivy::{
    directory::MmapDirectory,
    query::{BooleanQuery, Query},
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, FAST, STORED,
    },
    Index, IndexReader, ReloadPolicy, Searcher, SingleSegmentIndexWriter, TantivyDocument, Term,
};

use crate::{directory::ram_directory_from_mmap_dir, tf_term_query::TfTermQuery};

const MAX_DF_RATIO: f32 = 0.15;

pub struct InvertedIndex {
    index: Index,
    writer: Option<SingleSegmentIndexWriter>,
    pub reader: IndexReader,
    token_cluster_id: Field,
    doc_id: Field,
    pending: HashMap<u64, Vec<(String, f32)>>,
}

impl InvertedIndex {
    pub fn open() -> Result<Self> {
        let directory_path = Self::default_directory_path();
        fs::create_dir_all(&directory_path)?;
        let directory = MmapDirectory::open(&directory_path)?;
        let (schema, token_cluster_id, doc_id) = Self::build_schema()?;
        let index = Index::open_or_create(directory, schema)?;
        let mem_budget = 500_000_000; // 500 MB heap
        let writer = SingleSegmentIndexWriter::new(index.to_owned(), mem_budget)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(Self {
            index,
            writer: Some(writer),
            reader,
            token_cluster_id,
            doc_id,
            pending: HashMap::new(),
        })
    }

    pub fn open_with_copy_from_disk_to_ram_directory() -> Result<Self> {
        let directory_path = Self::default_directory_path();
        let (schema, token_cluster_id, doc_id) = Self::build_schema()?;
        let ram_index = if directory_path.exists() {
            let ram_directory = ram_directory_from_mmap_dir(&directory_path)?;
            Index::open(ram_directory)?
        } else {
            Index::create_in_ram(schema)
        };
        let reader = ram_index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(Self {
            index: ram_index,
            writer: None,
            reader,
            token_cluster_id,
            doc_id,
            pending: HashMap::new(),
        })
    }

    fn default_directory_path() -> PathBuf {
        std::env::var_os("SPARKBERT_INVERTED_INDEX_DIR")
            .map(PathBuf::from)
            .context("Please set SPARKBERT_INVERTED_INDEX_DIR env variable")
            .unwrap()
    }

    fn build_schema() -> Result<(Schema, Field, Field)> {
        let mut schema_builder = Schema::builder();
        let tok_opts = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::WithFreqs),
        );
        let token_cluster_id = schema_builder.add_text_field("token", tok_opts);
        let doc_id = schema_builder.add_u64_field("doc_id", FAST | STORED);
        let schema = schema_builder.build();
        Ok((schema, token_cluster_id, doc_id))
    }

    pub fn index(&mut self, doc_id: u64, tokens: Vec<String>, scores: Vec<f32>) {
        debug_assert_eq!(tokens.len(), scores.len());
        let doc_entry = self.pending.entry(doc_id).or_default();
        for (token, score) in tokens.into_iter().zip(scores.into_iter()) {
            doc_entry.push((token, score));
        }
    }

    /// commit
    pub fn finalize(&mut self) -> Result<()> {
        let mut writer = self.writer.take().unwrap();
        let stop_words = self.prepare_stop_words();
        for (&doc_id, token_score_pairs) in self.pending.iter() {
            let mut doc = TantivyDocument::new();
            doc.add_u64(self.doc_id, doc_id);
            let mut set = false;
            for (token, score) in token_score_pairs {
                if stop_words.contains(token) {
                    continue;
                }
                // TODO: remove magic number, use stats
                if *score < 22.7136 {
                    continue;
                }
                // TODO: add boundaries and try without f8
                let reps = F8E4M3::from_f32(*score).to_bits();
                if reps == 0 {
                    continue;
                }
                set = true;
                // set score as tf
                for _ in 0..reps {
                    doc.add_text(self.token_cluster_id, token);
                }
            }
            if set {
                writer.add_document(doc)?;
            } else {
                panic!("adjust hyperparams, no tokens were added to doc")
            }
        }
        self.pending.clear();
        println!("Inverted index memory usage: {}", &writer.mem_usage());
        let index = writer.finalize()?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        self.index = index;
        self.reader = reader;
        Ok(())
    }

    fn prepare_stop_words(&self) -> HashSet<&String> {
        let mut token_to_doc_count = HashMap::new();
        for (_, token_score_pairs) in self.pending.iter() {
            let mut seen = HashSet::new();
            for (token, _) in token_score_pairs {
                if seen.insert(token) {
                    *token_to_doc_count.entry(token).or_insert(0) += 1;
                }
            }
        }
        let total_docs = self.pending.len() as f32;
        token_to_doc_count
            .into_iter()
            .filter(|(_, doc_count)| (*doc_count as f32 / total_docs) >= MAX_DF_RATIO)
            .map(|(token, _)| token)
            .collect()
    }

    pub fn get_num_docs(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }

    // TODO: 1. построить графики качество/время 2. посмотреть на глубину обхода постинг листов
    /// execute a query that is a list of `(token#cluster)` strings
    /// returns `Vec<(doc_id, sum_score)>` sorted desc by sum_score
    pub fn search(
        &self,
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
            &self.reader.searcher()
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
