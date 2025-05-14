mod args;
mod dataset;
mod embs;
mod inverted_index;
mod postings;
mod run;
mod score;
mod util;
mod vector_index;
use anyhow::Result;
use dataset::load_scifact;
use faiss::read_index;
use faiss::Index;
use postings::build_postings;
use redis::Commands;
use run::find_tokens;
use run::load_inverted_index;
use std::collections::HashMap;
use util::device;

#[tokio::main]
async fn main() -> Result<()> {
    let mut redis = redis::Client::open("redis://cache.home:16379")?.get_connection()?;
    let faiss_idx_to_token: HashMap<String, String> = redis.hgetall("faiss_idx_to_token")?;
    let mut vector_index = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    println!("Vector dictionary size: {}", vector_index.ntotal());
    let d = vector_index.d() as usize;
    let device = device(false)?;
    let index_n_neighbors = 8;
    let search_n_neighbors = 3;
    let search_top_k = 10;
    let (corpus, _queries, _qrelss) = load_scifact("test")?;
    if !redis.exists("postings")? {
        build_postings(
            &corpus,
            &mut vector_index,
            &faiss_idx_to_token,
            index_n_neighbors,
            &mut redis,
            d,
            &device,
        )
        .await?;
    }
    let mut inverted_index = load_inverted_index(&mut redis)?;
    let query = "some test query";
    let tokens = find_tokens(
        &mut vector_index,
        &search_n_neighbors,
        &faiss_idx_to_token,
        query,
    )?;
    let results = inverted_index.search(tokens.as_slice(), search_top_k)?;
    println!("Search results: {:?}", results);
    Ok(())
}
