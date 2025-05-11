mod args;
mod dataset;
mod embs;
mod inverted_index;
mod postings;
mod score;
mod util;
mod vector_index;
use anyhow::Result;
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

/// ---------- demo ---------------------------------------------
#[tokio::main]
async fn main() -> Result<()> {
    let mut redis = redis::Client::open("redis://cache.home:16379")?.get_connection()?;
    let faiss_idx_to_token: HashMap<String, String> = redis.hgetall("faiss_idx_to_token")?;
    let mut index = read_index("/home/slava/Developer/SparKBERT/hnsw.index")?;
    println!("Vector dictionary size: {}", index.ntotal());
    let d = index.d() as usize;
    let device = device(false)?;
    let index_n_neighbors = 8;
    let (corpus, queries, qrels) = load_scifact("test")?;
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
    let query_pairs = vec!["9733_6".to_string()];
    inverted_index.search(&query_pairs, 10)?;
    Ok(())
}
