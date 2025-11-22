mod api;
mod args;
mod dataset;
mod directory;
mod embs;
mod indexing;
mod inverted_index;
mod score;
mod tf_term_query;
mod util;
mod vector_vocabulary;
use anyhow::Result;
use api::Config;
use api::SparkBert;
use dataset::load_scifact;
use indexing::build_spark_bert;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use util::device;
use util::get_progress_bar;

#[tokio::main]
async fn main() -> Result<()> {
    let device = device(false)?;
    let index_n_neighbors = 8;
    let search_n_neighbors = 3;
    let search_top_k = 1000;
    let (corpus, queries, _qrels) = load_scifact("test")?;
    let config = Config {
        use_ram_index: false,
        device: device.to_owned(),
        index_n_neighbors,
    };
    let mut spark_bert = SparkBert::new(config)?;
    if spark_bert.get_num_docs() == 0 {
        drop(spark_bert);
        spark_bert = build_spark_bert(&corpus, index_n_neighbors, device)?;
    }
    println!("SparkBERT index size: {}", spark_bert.get_num_docs());
    let pb = get_progress_bar(queries.len() as u64)?;
    let mut results: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (query_id, query) in pb.wrap_iter(queries.into_iter()) {
        let doc_id_score_pairs =
            spark_bert.search(&query.text, search_n_neighbors, search_top_k)?;
        let mut query_results = HashMap::new();
        for (doc_id, score) in doc_id_score_pairs {
            query_results.insert(doc_id.to_string(), score);
        }
        results.insert(query_id, query_results);
    }
    let results_json = serde_json::to_string_pretty(&results)?;
    let results_path = std::env::current_dir()?.join("results.json");
    let mut results_file = File::create(results_path)?;
    results_file.write_all(results_json.as_bytes())?;
    Ok(())
}
