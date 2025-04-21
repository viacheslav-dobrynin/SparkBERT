mod args;
mod embs;
mod inverted_index;
mod util;
mod vector_index;
use anyhow::Result;
use faiss::read_index;
use inverted_index::InvertedIndex;

/// ---------- demo ---------------------------------------------
fn main() -> Result<()> {
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

    read_index("../hnsw.index");

    Ok(())
}
