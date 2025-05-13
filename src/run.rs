use std::{collections::HashMap, time::Instant};

use anyhow::{Ok, Result};
use candle_core::D;
use faiss::Index;
use redis::{Commands, Connection};

use crate::{embs::calc_embs, inverted_index::InvertedIndex, postings::ArchivedPosting};

pub fn load_inverted_index(redis: &mut Connection) -> Result<InvertedIndex> {
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
    let flat_embs = query_embs.flatten_all()?.to_vec1::<f32>()?;
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
