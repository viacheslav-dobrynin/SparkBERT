use std::collections::HashMap;

use anyhow::Result;
use candle_core::Device;
use faiss::index::IndexImpl;
use faiss::Index;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::Condition;
use qdrant_client::qdrant::Filter;
use qdrant_client::qdrant::ScrollPointsBuilder;
use qdrant_client::Qdrant;
use redis::Commands;
use redis::Connection;
use rkyv::{rancor::Error, Archive, Deserialize, Serialize};

use crate::dataset::CorpusDoc;
use crate::score::calculate_max_sim;
use crate::util::get_progress_bar;
use crate::vector_index::reconstruct_batch;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(
    // This will generate a PartialEq impl between our unarchived
    // and archived types
    compare(PartialEq),
    // Derives can be passed through to the generated type:
    derive(Debug),
)]
pub struct Posting {
    pub token: String,
    pub doc_id: u64,
    pub score: f32,
}

async fn load_embs_for_doc_id(client: &Qdrant, d: usize, doc_id: &str) -> Result<Vec<f32>> {
    let scroll_response = client
        .scroll(
            ScrollPointsBuilder::new("scifact_embs_all_MiniLM_L6_v2")
                .filter(Filter::must([Condition::matches(
                    "doc_id",
                    doc_id.to_string(),
                )]))
                .limit(1000)
                .with_payload(false)
                .with_vectors(true),
        )
        .await?;

    // Pre-calculate capacity to avoid reallocations
    let point_count = scroll_response.result.len();
    if point_count >= 999 {
        return Err(anyhow::anyhow!(
            "Too many points returned for document {}",
            doc_id
        ));
    }

    let mut flat_embs: Vec<f32> = Vec::with_capacity(point_count * d);

    for point in scroll_response.result {
        let vectors_options = point.vectors.unwrap().vectors_options.unwrap();
        match vectors_options {
            VectorsOptions::Vector(vector_output) => {
                flat_embs.extend(&vector_output.data);
            }
            VectorsOptions::Vectors(_named_vectors_output) => {
                panic!("Unexpected vectors for token")
            }
        }
    }
    Ok(flat_embs)
}

pub async fn build_postings(
    corpus: &HashMap<String, CorpusDoc>,
    index: &mut IndexImpl,
    faiss_idx_to_token: &HashMap<String, String>,
    index_n_neighbors: usize,
    redis: &mut Connection,
    device: &Device,
) -> Result<()> {
    let qdrant = Qdrant::from_url("http://vectordb.home:6334").build()?;
    let pb = get_progress_bar(corpus.len() as u64)?;
    let d = index.d() as usize;
    for (doc_id, _doc) in pb.wrap_iter(corpus.iter()) {
        let doc_embs = load_embs_for_doc_id(&qdrant, d, doc_id).await?;
        let faiss::index::SearchResult {
            distances: _,
            labels,
        } = index.search(&doc_embs, index_n_neighbors)?;
        let token_embs = reconstruct_batch(index, &labels)?;
        let tokens: Vec<&String> = labels
            .iter()
            .map(|idx| {
                let idx = idx.get().unwrap().to_string();
                faiss_idx_to_token.get(&idx).unwrap()
            })
            .collect();
        let scores = calculate_max_sim(doc_embs, token_embs, device, d)?;
        debug_assert!(tokens.len() == scores.len());
        let mut batch: Vec<Vec<u8>> = Vec::with_capacity(1_000);
        for (token, score) in tokens.iter().zip(scores) {
            let posting = Posting {
                token: token.to_string(),
                doc_id: doc_id.parse::<u64>()?,
                score,
            };
            batch.push(rkyv::to_bytes::<Error>(&posting).unwrap().to_vec());

            if batch.len() == 1_000 {
                let _: () = redis.rpush("postings", &batch)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            let _: () = redis.rpush("postings", &batch)?;
        }
    }
    Ok(())
}
