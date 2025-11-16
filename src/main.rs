mod args;
mod dataset;
mod embs;
mod indexing;
mod inverted_index;
mod postings;
mod run;
mod score;
mod tf_term_query;
mod util;
mod vector_index;
use anyhow::Result;
use dataset::load_scifact;
use faiss::read_index;
use faiss::Index;
use indexing::build_inverted_index;
use postings::build_postings;
use redis::Commands;
use run::find_tokens;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use util::device;
use util::get_progress_bar;
use vector_index::load_faiss_idx_to_token;

#[tokio::main]
async fn main() -> Result<()> {
    let mut redis = redis::Client::open("redis://cache.home:16379")?.get_connection()?;
    let faiss_idx_to_token: HashMap<String, String> =
        load_faiss_idx_to_token("/home/slava/Developer/SparKBERT/faiss_idx_to_token.json")?;
    let mut vector_vocabulary = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    println!("Vector vocabulary size: {}", vector_vocabulary.ntotal());
    let device = device(false)?;
    let index_n_neighbors = 8;
    let search_n_neighbors = 3;
    let search_top_k = 1000;
    let (corpus, queries, _qrels) = load_scifact("test")?;
    if !redis.exists("postings")? {
        build_postings(
            &corpus,
            &mut vector_vocabulary,
            &faiss_idx_to_token,
            index_n_neighbors,
            &mut redis,
            &device,
        )
        .await?;
    }
    // let inverted_index = load_inverted_index(&mut redis)?;
    let inverted_index = build_inverted_index(
        &corpus,
        &mut vector_vocabulary,
        &faiss_idx_to_token,
        index_n_neighbors,
        &device,
    )?;
    let pb = get_progress_bar(queries.len() as u64)?;
    let mut results: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (query_id, query) in pb.wrap_iter(queries.into_iter()) {
        let tokens = find_tokens(
            &mut vector_vocabulary,
            &search_n_neighbors,
            &faiss_idx_to_token,
            &query.text,
        )?;
        let doc_id_score_pairs = inverted_index.search(None, tokens.as_slice(), search_top_k)?;
        let mut query_results = HashMap::new();
        for (doc_id, score) in doc_id_score_pairs {
            query_results.insert(doc_id.to_string(), score);
        }
        results.insert(query_id, query_results);
    }
    let results_json = serde_json::to_string_pretty(&results)?;
    let mut results_file = File::create("/home/slava/Developer/SparKBERT/results.json")?;
    results_file.write_all(results_json.as_bytes())?;
    Ok(())
}
