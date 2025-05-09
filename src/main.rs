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
use inverted_index::InvertedIndex;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::Condition;
use qdrant_client::qdrant::Filter;
use qdrant_client::qdrant::Match;
use qdrant_client::qdrant::ScrollPointsBuilder;
use qdrant_client::Qdrant;
use redis::Commands;
use std::collections::HashMap;
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

fn build_inverted_index() -> Result<()> {
    let mut idx = InvertedIndex::open()?;

    // ---------- indexing ----------
    // business doc A has 2 non‑zero pairs
    idx.add_pair("42#7", 1, 0.9)?;
    idx.add_pair("13#2", 1, 0.4)?;
    idx.add_pair("42#7", 2, 0.8)?;
    idx.add_pair("11#1", 2, 0.6)?;
    idx.add_pair("99#5", 2, 0.1)?;

    idx.commit()?; // force merge & commit

    // ---------- search ------------
    let query_pairs = vec!["42#7".to_string(), "11#1".to_string()];
    idx.search(&query_pairs, 10)?;
    Ok(())
}

async fn load_embs_for_doc_id(client: &Qdrant, doc_id: &str) -> Result<Vec<f32>> {
    let scroll_response = client
        .scroll(
            ScrollPointsBuilder::new("scifact_embs_all_MiniLM_L6_v2")
                .filter(Filter::must([Condition::matches(
                    "doc_id",
                    doc_id.to_string(),
                )]))
                .limit(1000)
                .with_payload(true)
                .with_vectors(true),
        )
        .await?;
    debug_assert!(scroll_response.result.len() < 999);
    let mut flat_embs: Vec<f32> = Vec::new();
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
    let qdrant = Qdrant::from_url("http://vectordb.home:6334").build()?;
    let mut redis = redis::Client::open("redis://cache.home:16379")?.get_connection()?;
    let faiss_idx_to_token: HashMap<String, String> = redis.hgetall("faiss_idx_to_token")?;
    let mut index = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    println!("Vector dictionary size: {}", index.ntotal());
    let (corpus, queries, qrels) = load_scifact("test")?;
    let number = 1;
    for (doc_id, doc) in &corpus {
        let flat_embs = load_embs_for_doc_id(&qdrant, doc_id).await?;
        dbg!(flat_embs.len());
        let faiss::index::SearchResult { distances, labels } = index.search(&flat_embs, 8)?;
        let embs = reconstruct_batch(&index, &labels)?;
        let tokens: Vec<&String> = labels
            .iter()
            .map(|idx| {
                let idx = idx.get().unwrap().to_string();
                faiss_idx_to_token.get(&idx).unwrap()
            })
            .collect();
        dbg!(tokens);
        if number == 1 {
            break;
        }
    }
    Ok(())
}
