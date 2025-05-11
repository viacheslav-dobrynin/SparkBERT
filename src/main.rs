mod args;
mod dataset;
mod embs;
mod inverted_index;
mod util;
mod vector_index;
use anyhow::Result;
use candle_core::Tensor;
use dataset::load_scifact;
use embs::calc_embs;
use faiss::read_index;
use faiss::Index;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use inverted_index::InvertedIndex;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::Condition;
use qdrant_client::qdrant::Filter;
use qdrant_client::qdrant::Match;
use qdrant_client::qdrant::ScrollPointsBuilder;
use qdrant_client::Qdrant;
use redis::Commands;
use std::collections::HashMap;
use std::fmt::Write;
use std::time::Instant;
use util::device;
use util::reconstruct_batch;

fn process_batch(
    ids: &[&String],
    texts: &[String],
    index: &mut faiss::index::IndexImpl,
) -> Result<()> {
    let embs = calc_embs(texts.iter().map(String::as_str).collect())?;
    let flat_embs = embs.flatten_all()?.to_vec1::<f32>()?;
    let faiss::index::SearchResult { distances, labels } = index.search(&flat_embs, 8)?;
    debug_assert_eq!(labels.len() / 8, embs.dim(0)?);
    let embs = reconstruct_batch(index, &labels)?;
    Ok(())
}

fn do_smtg() -> Result<()> {
    let mut index = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    println!("{}", index.ntotal());

    let batch = 128 / 4;
    let mut ids = Vec::with_capacity(batch);
    let mut texts = Vec::with_capacity(batch);
    let (corpus, queries, qrels) = load_scifact("test")?;
    for (id, doc) in &corpus {
        let text = if let Some(title) = &doc.title {
            // одна аллокация, чтобы не делать format! для None
            let mut s = String::with_capacity(title.len() + 1 + doc.text.len());
            s.push_str(title);
            s.push(' ');
            s.push_str(&doc.text);
            s
        } else {
            doc.text.clone()
        };

        ids.push(id); // храним только ссылки на id
        texts.push(text);
        if texts.len() == batch {
            process_batch(&ids, &texts, &mut index)?;
            ids.clear();
            texts.clear();
        }
    }
    if !texts.is_empty() {
        process_batch(&ids, &texts, &mut index)?;
    }
    println!("{}, {}, {}", corpus.len(), queries.len(), qrels.len());
    Ok(())
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
            VectorsOptions::Vectors(named_vectors_output) => {
                panic!("Unexpected vectors for token")
            }
        }
    }
    Ok(flat_embs)
}

/// ---------- demo ---------------------------------------------
#[tokio::main]
async fn main() -> Result<()> {
    let mut inverted_index = InvertedIndex::open()?;
    let qdrant = Qdrant::from_url("http://vectordb.home:6334").build()?;
    let mut redis = redis::Client::open("redis://cache.home:16379")?.get_connection()?;
    let faiss_idx_to_token: HashMap<String, String> = redis.hgetall("faiss_idx_to_token")?;
    let mut index = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    let d = index.d() as usize;
    let device = device(false)?;
    let index_n_neighbors = 8;
    println!("Vector dictionary size: {}", index.ntotal());
    let (corpus, queries, qrels) = load_scifact("test")?;
    let pb = ProgressBar::new(corpus.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    for (doc_id, doc) in pb.wrap_iter(corpus.iter()) {
        let doc_embs = load_embs_for_doc_id(&qdrant, d, doc_id).await?;
        let faiss::index::SearchResult {
            distances: _,
            labels,
        } = index.search(&doc_embs, index_n_neighbors)?;
        let token_embs = reconstruct_batch(&index, &labels)?;
        let tokens: Vec<&String> = labels
            .iter()
            .map(|idx| {
                let idx = idx.get().unwrap().to_string();
                faiss_idx_to_token.get(&idx).unwrap()
            })
            .collect();
        let doc_embs_count = doc_embs.len() / d;
        let doc_embs_tensor = Tensor::from_vec(doc_embs, (doc_embs_count, d), &device)?;
        let token_embs_count = token_embs.len() / d;
        let token_embs_tensor = Tensor::from_vec(token_embs, (token_embs_count, d), &device)?;
        let scores: Vec<f32> = doc_embs_tensor
            .matmul(&token_embs_tensor.t()?)?
            .max(0)?
            .to_vec1()?;
        debug_assert!(tokens.len() == scores.len());
        for (token, score) in tokens.iter().zip(scores.iter()) {
            inverted_index.add_pair(token, doc_id.parse::<u64>()?, *score as f64)?;
        }
    }
    inverted_index.commit()?;
    let query_pairs = vec!["9733_6".to_string()];
    let start = Instant::now();
    inverted_index.search(&query_pairs, 10)?;
    dbg!(start.elapsed());
    Ok(())
}
