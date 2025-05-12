mod args;
mod dataset;
mod embs;
mod inverted_index;
mod postings;
mod score;
mod util;
mod vector_index;
use anyhow::Result;
use candle_core::D;
use dataset::load_scifact;
use embs::calc_embs;
use faiss::read_index;
use faiss::Index;
use inverted_index::InvertedIndex;
use postings::build_postings;
use postings::ArchivedPosting;
use redis::Commands;
use std::collections::HashMap;
use std::time::Instant;
use util::device;

#[tokio::main]
async fn main() -> Result<()> {
    let mut redis = redis::Client::open("redis://cache.home:16379")?.get_connection()?;
    let faiss_idx_to_token: HashMap<String, String> = redis.hgetall("faiss_idx_to_token")?;
    let mut index = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    println!("Vector dictionary size: {}", index.ntotal());
    let d = index.d() as usize;
    let device = device(false)?;
    let index_n_neighbors = 8;
    let search_n_neighbors = 3;
    let search_top_k = 10;
    let (corpus, _queries, _qrelss) = load_scifact("test")?;
    if !redis.exists("postings")? {
        build_postings(
            &corpus,
            &mut index,
            &faiss_idx_to_token,
            index_n_neighbors,
            &mut redis,
            d,
            &device,
        )
        .await?;
    }
    let inverted_index_building_start = Instant::now();
    let mut inverted_index = InvertedIndex::open()?;
    let postings: Vec<Vec<u8>> = redis.lrange("postings", 0, -1)?;
    for raw in postings {
        let posting = unsafe { rkyv::access_unchecked::<ArchivedPosting>(raw.as_slice()) };
        let token = posting.token.as_str();
        let doc_id: u64 = posting.doc_id.into();
        let score32: f32 = posting.score.into();
        let score = score32 as f64;
        inverted_index.add_pair(token, doc_id, score)?;
    }
    inverted_index.commit()?;
    dbg!(inverted_index_building_start.elapsed());

    let query = "some test query";
    let search_start = Instant::now();
    let query_embs = calc_embs(vec![query], false)?;
    let flat_embs = query_embs.flatten_all()?.to_vec1::<f32>()?;
    let faiss::index::SearchResult {
        distances: _,
        labels,
    } = index.search(&flat_embs, search_n_neighbors)?;
    let tokens: Vec<&str> = labels
        .iter()
        .map(|idx| {
            let idx = idx.get().unwrap().to_string();
            faiss_idx_to_token.get(&idx).map(String::as_str).unwrap()
        })
        .collect();
    debug_assert_eq!(
        labels.len() / search_n_neighbors,
        query_embs.dim(D::Minus2)?
    );
    inverted_index.search(tokens.as_slice(), search_top_k)?;
    dbg!(search_start.elapsed());
    Ok(())
}
