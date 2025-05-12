use anyhow::Result;
use serde_json::json;
use tantivy::{
    aggregation::{agg_req::Aggregations, AggregationCollector},
    doc,
    query::{BooleanQuery, TermQuery},
    schema::{Field, IndexRecordOption, Schema, FAST, STORED, STRING},
    Index, IndexWriter, ReloadPolicy, Term,
};

/// ------ user‑facing wrapper ----------------------------------------------
pub struct InvertedIndex {
    index: Index,
    writer: IndexWriter,
    // cached field handles
    token_cluster_id: Field,
    doc_id: Field,
    score: Field,
}

impl InvertedIndex {
    pub fn open() -> Result<Self> {
        // ---- schema -----
        let mut schema_builder = Schema::builder();
        let token_cluster_id = schema_builder.add_text_field("token_cluster_id", STRING);
        let doc_id = schema_builder.add_u64_field("doc_id", FAST | STORED);
        let score = schema_builder.add_f64_field("score", FAST);
        let schema = schema_builder.build();

        let index = Index::builder().schema(schema).create_in_ram()?;

        let writer = index.writer(50_000_000)?; // 50 MB heap

        Ok(Self {
            index,
            writer,
            token_cluster_id,
            doc_id,
            score,
        })
    }

    /// add one (token,cluster) pair as a sub‑document
    pub fn add_pair(&mut self, token_cluster_id: &str, doc_id: u64, score: f64) -> Result<()> {
        self.writer.add_document(doc!(
            self.token_cluster_id => token_cluster_id,
            self.doc_id => doc_id,
            self.score => score,
        ))?;
        Ok(())
    }

    /// commit
    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        Ok(())
    }

    /// execute a query that is a list of `(token#cluster)` strings
    /// returns `Vec<(doc_gid, sum_score)>` sorted desc by sum_score
    pub fn search(&self, pairs: &[&str], top_k: usize) -> Result<()> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();

        // Boolean OR over all given pairs
        let term_queries = pairs
            .iter()
            .map(|s| {
                TermQuery::new(
                    Term::from_field_text(self.token_cluster_id, s),
                    IndexRecordOption::Basic,
                )
            })
            .map(|q| Box::new(q) as Box<dyn tantivy::query::Query>)
            .collect();

        let bool_query = BooleanQuery::union(term_queries); // :contentReference[oaicite:0]{index=0}

        // Aggregation: group by gid, sum psc, order by sum desc
        let aggs: Aggregations = serde_json::from_value(
            json!({                                   // :contentReference[oaicite:1]{index=1}
                "by_doc_id": {
                    "terms": { "field": "doc_id", "order": { "sum_score": "desc" }, "size": top_k },
                    "aggs": { "sum_score": { "sum": { "field": "score" } } }
                }
            }),
        )?;

        let collector = AggregationCollector::from_aggs(aggs, Default::default());

        let res = searcher.search(&bool_query, &collector)?;

        //println!("{res:#?}");
        Ok(())
    }
}
