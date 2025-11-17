use std::collections::HashMap;

use anyhow::Result;
use candle_core::D;
use faiss::Index;

use crate::embs::{calc_embs, convert_to_flatten_vec};

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
