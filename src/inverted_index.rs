use anyhow::Result;
use serde_json::json;
use tantivy::{
    aggregation::{
        agg_req::Aggregations,
        agg_result::{AggregationResult, BucketResult},
        AggregationCollector,
    },
    directory::RamDirectory,
    doc,
    query::{BooleanQuery, TermQuery},
    schema::{Field, IndexRecordOption, Schema, FAST, STORED, STRING},
    Index, IndexReader, ReloadPolicy, Searcher, SingleSegmentIndexWriter, Term,
};

pub struct InvertedIndex {
    index: Option<Index>,
    writer: Option<SingleSegmentIndexWriter>,
    pub reader: Option<IndexReader>,
    collector_by_top_k: Option<(usize, AggregationCollector)>,
    token_cluster_id: Field,
    doc_id: Field,
    score: Field,
}

impl InvertedIndex {
    pub fn open() -> Result<Self> {
        let mut schema_builder = Schema::builder();
        let token_cluster_id = schema_builder.add_text_field("token_cluster_id", STRING);
        let doc_id = schema_builder.add_u64_field("doc_id", FAST | STORED);
        let score = schema_builder.add_f64_field("score", FAST);
        let schema = schema_builder.build();

        let writer = Index::builder()
            .schema(schema)
            .single_segment_index_writer(RamDirectory::create(), 50_000_000)?; // 50 MB heap
        let writer = Some(writer);

        Ok(Self {
            index: None,
            writer,
            reader: None,
            collector_by_top_k: None,
            token_cluster_id,
            doc_id,
            score,
        })
    }

    /// add one (token,cluster) pair as a sub‑document
    pub fn add_pair(&mut self, token_cluster_id: &str, doc_id: u64, score: f64) -> Result<()> {
        self.writer.as_mut().unwrap().add_document(doc!(
            self.token_cluster_id => token_cluster_id,
            self.doc_id => doc_id,
            self.score => score,
        ))?;
        Ok(())
    }

    /// commit
    pub fn finalize(&mut self) -> Result<()> {
        let writer = self.writer.take().unwrap();
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

    pub fn ensure_collector(&mut self, top_k: usize) -> Result<&AggregationCollector> {
        if self.collector_by_top_k.is_none() || self.collector_by_top_k.as_ref().unwrap().0 != top_k
        {
            let aggs: Aggregations = serde_json::from_value(json!({
                "by_doc_id": {
                    "terms": {
                        "field": "doc_id",
                        "order": { "sum_score": "desc" },
                        "size": top_k
                    },
                    "aggs": {
                        "sum_score": {
                            "sum": { "field": "score" }
                        }
                    }
                }
            }))?;
            let collector = AggregationCollector::from_aggs(aggs, Default::default());
            self.collector_by_top_k = Some((top_k, collector));
        }
        Ok(&self.collector_by_top_k.as_ref().unwrap().1)
    }

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
            let reader = self.reader.as_ref().unwrap();
            &reader.searcher()
        };

        let mut term_queries = Vec::with_capacity(pairs.len());

        for token_cluster_id in pairs {
            let term = Term::from_field_text(self.token_cluster_id, token_cluster_id);
            let query = TermQuery::new(term, IndexRecordOption::Basic);
            term_queries.push(Box::new(query) as Box<dyn tantivy::query::Query>);
        }

        let bool_query = BooleanQuery::union(term_queries);

        let collector = self.ensure_collector(top_k)?;
        let agg_result = searcher.search(&bool_query, collector)?;

        let mut results = Vec::new();

        if let Some(by_doc_id) = agg_result.0.get("by_doc_id") {
            match by_doc_id {
                AggregationResult::BucketResult(bucker_result) => match bucker_result {
                    BucketResult::Terms {
                        buckets,
                        sum_other_doc_count: _,
                        doc_count_error_upper_bound: _,
                    } => {
                        for bucket in buckets {
                            let doc_id = match bucket.key {
                                tantivy::aggregation::Key::Str(_) => todo!(),
                                tantivy::aggregation::Key::I64(_) => todo!(),
                                tantivy::aggregation::Key::U64(key) => key,
                                tantivy::aggregation::Key::F64(_) => todo!(),
                            };
                            let sum_score = bucket.sub_aggregation.0.get("sum_score").unwrap();
                            let score =match sum_score {
                                AggregationResult::BucketResult(_bucket_result) => todo!(),
                                AggregationResult::MetricResult(metric_result) => {
                                    match metric_result {
                                        tantivy::aggregation::agg_result::MetricResult::Average(_single_metric_result) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::Count(_single_metric_result) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::Max(_single_metric_result) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::Min(_single_metric_result) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::Stats(_stats) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::ExtendedStats(_extended_stats) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::Sum(single_metric_result) => {
                                            single_metric_result.value.unwrap()
                                        },
                                        tantivy::aggregation::agg_result::MetricResult::Percentiles(_percentiles_metric_result) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::TopHits(_top_hits_metric_result) => todo!(),
                                        tantivy::aggregation::agg_result::MetricResult::Cardinality(_single_metric_result) => todo!(),
                                    }
                                },
                            };
                            results.push((doc_id, score));
                        }
                    }
                    BucketResult::Histogram { buckets: _ } => todo!(),
                    BucketResult::Range { buckets: _ } => todo!(),
                },
                AggregationResult::MetricResult(_metric_result) => todo!(),
            }
        }

        Ok(results)
    }
}
