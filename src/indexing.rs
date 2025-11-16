use std::{collections::HashMap, time::Instant, vec};

use anyhow::Result;
use candle_core::Device;
use faiss::{index::SearchResult, Index};

use crate::{
    dataset::Corpus,
    embs::{calc_embs, convert_to_flatten_vec},
    inverted_index::InvertedIndex,
    score::calculate_max_sim,
    util::get_progress_bar,
    vector_index::{reconstruct_batch, unique_labels},
};

pub fn build_inverted_index<T>(
    corpus: &Corpus,
    vector_vocabulary: &mut T,
    faiss_idx_to_token: &HashMap<String, String>,
    index_n_neighbors: usize,
    device: &Device,
) -> Result<InvertedIndex>
where
    T: Index + Sync,
{
    let inverted_index_building_start = Instant::now();
    let mut inverted_index = InvertedIndex::open()?;
    let pb = get_progress_bar(corpus.len() as u64)?;
    let d = vector_vocabulary.d() as usize;
    for (doc_id_string, doc) in pb.wrap_iter(corpus.iter()) {
        let doc_id: u64 = doc_id_string
            .parse()
            .expect("Failed to parse string to u64");
        let doc_repr = doc.as_text();
        let doc_embs = convert_to_flatten_vec(&calc_embs(vec![doc_repr.as_str()], false)?)?;
        let SearchResult {
            distances: _,
            labels,
        } = vector_vocabulary.search(&doc_embs, index_n_neighbors)?;
        let labels = unique_labels(&labels);
        let tokens: Vec<&String> = labels
            .iter()
            .map(|idx| {
                let idx = idx.get().unwrap().to_string();
                faiss_idx_to_token.get(&idx).unwrap()
            })
            .collect();
        let token_embs = reconstruct_batch(vector_vocabulary, &labels)?;
        let scores = calculate_max_sim(doc_embs, token_embs, device, d)?;
        for (&token, &score) in tokens.iter().zip(scores.iter()) {
            inverted_index.add_pair(token, doc_id, score)?;
        }
    }
    inverted_index.finalize()?;
    dbg!(inverted_index_building_start.elapsed());
    Ok(inverted_index)
}
