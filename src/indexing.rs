use std::time::Instant;

use anyhow::Result;
use candle_core::Device;

use crate::{
    dataset::Corpus,
    embs::{calc_embs, convert_to_flatten_vec},
    inverted_index::InvertedIndex,
    score::calculate_max_sim,
    util::get_progress_bar,
    vector_vocabulary::VectorVocabulary,
};

pub fn build_inverted_index(
    vector_vocabulary: &mut VectorVocabulary,
    corpus: &Corpus,
    index_n_neighbors: usize,
    device: &Device,
) -> Result<InvertedIndex> {
    let inverted_index_building_start = Instant::now();
    let mut inverted_index = InvertedIndex::open()?;
    let pb = get_progress_bar(corpus.len() as u64)?;
    let d = vector_vocabulary.get_embedding_dims() as usize;
    for (doc_id_string, doc) in pb.wrap_iter(corpus.iter()) {
        let doc_id: u64 = doc_id_string
            .parse()
            .expect("Failed to parse string to u64");
        let doc_repr = doc.as_text();
        let doc_embs = convert_to_flatten_vec(&calc_embs(vec![doc_repr.as_str()], false)?)?;
        let (tokens, token_embs) =
            vector_vocabulary.find_tokens(&doc_embs, index_n_neighbors, true)?;
        let scores = calculate_max_sim(doc_embs, token_embs.unwrap(), device, d)?;
        inverted_index.index(
            doc_id,
            tokens.iter().map(|&it| it.to_owned()).collect(),
            scores,
        );
    }
    inverted_index.finalize()?;
    dbg!(inverted_index_building_start.elapsed());
    Ok(inverted_index)
}
