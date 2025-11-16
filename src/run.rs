use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use anyhow::{Ok, Result};
use candle_core::D;
use faiss::Index;
use redis::{Commands, Connection};

use crate::{
    embs::{calc_embs, convert_to_flatten_vec},
    inverted_index::InvertedIndex,
};

const MAX_DF_RATIO: f64 = 0.15;

pub fn load_inverted_index(redis: &mut Connection) -> Result<InvertedIndex> {
    let inverted_index_building_start = Instant::now();
    let raw_postings: Vec<String> = redis.lrange("postings_py", 0, -1)?;
    let mut per_token_docs: HashMap<String, HashSet<u64>> = HashMap::new();

    for raw in &raw_postings {
        let mut it = raw.splitn(3, '|');
        let token = it.next().unwrap().to_string();
        let doc_id = it.next().unwrap().parse()?;
        per_token_docs.entry(token).or_default().insert(doc_id);
    }
    let total_docs = per_token_docs
        .values()
        .flat_map(|hs| hs.iter().copied())
        .collect::<HashSet<u64>>()
        .len() as f64;
    let stop_words: HashSet<String> = per_token_docs
        .into_iter()
        .filter(|(_, docs)| (docs.len() as f64 / total_docs) >= MAX_DF_RATIO)
        .map(|(tok, _)| tok)
        .collect();

    let mut inverted_index = InvertedIndex::open()?;

    for raw in raw_postings {
        let mut it = raw.splitn(3, '|');
        let token = it.next().unwrap().to_string();
        if stop_words.contains(&token) {
            continue;
        }

        let doc_id = it.next().unwrap().parse()?;
        let score: f32 = it.next().unwrap().parse()?;
        if score >= 22.7136 {
            inverted_index.add_pair(&token, doc_id, score)?;
        }
    }
    inverted_index.finalize()?;
    dbg!(inverted_index_building_start.elapsed());
    Ok(inverted_index)
}

pub fn find_tokens<'a, T>(
    vector_index: &mut T,
    search_n_neighbors: &usize,
    faiss_idx_to_token: &'a HashMap<String, String>,
    query: &str,
) -> Result<Vec<&'a str>>
where
    T: Index + Sync,
{
    let query_embs = calc_embs(vec![query], false)?;
    let flat_embs = convert_to_flatten_vec(&query_embs)?;
    let faiss::index::SearchResult {
        distances: _,
        labels,
    } = vector_index.search(&flat_embs, *search_n_neighbors)?;
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
    Ok(tokens)
}
