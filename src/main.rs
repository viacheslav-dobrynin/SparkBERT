mod args;
mod dataset;
mod directory;
mod embs;
mod indexing;
mod inverted_index;
mod score;
mod tf_term_query;
mod util;
mod vector_index;
use anyhow::Result;
use dataset::load_scifact;
use embs::calc_embs;
use embs::convert_to_flatten_vec;
use indexing::build_inverted_index;
use inverted_index::InvertedIndex;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use util::device;
use util::get_progress_bar;
use vector_index::VectorVocabulary;

#[tokio::main]
async fn main() -> Result<()> {
    let mut vector_vocabulary = VectorVocabulary::build()?;
    println!(
        "Vector vocabulary size: {}",
        vector_vocabulary.get_num_tokens()
    );
    let device = device(false)?;
    let index_n_neighbors = 8;
    let search_n_neighbors = 3;
    let search_top_k = 1000;
    let (corpus, queries, _qrels) = load_scifact("test")?;
    let mut inverted_index = InvertedIndex::open_with_copy_from_disk_to_ram_directory()?;
    if inverted_index.get_num_docs()? == 0 {
        inverted_index =
            build_inverted_index(&mut vector_vocabulary, &corpus, index_n_neighbors, &device)?;
    }
    println!("Inverted index size: {}", inverted_index.get_num_docs()?);
    let pb = get_progress_bar(queries.len() as u64)?;
    let mut results: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (query_id, query) in pb.wrap_iter(queries.into_iter()) {
        let query_embs = calc_embs(vec![&query.text], false)?;
        let flat_query_embs = convert_to_flatten_vec(&query_embs)?;
        let (tokens, _) =
            vector_vocabulary.find_tokens(&flat_query_embs, search_n_neighbors, false)?;
        let doc_id_score_pairs = inverted_index.search(None, tokens.as_slice(), search_top_k)?;
        let mut query_results = HashMap::new();
        for (doc_id, score) in doc_id_score_pairs {
            query_results.insert(doc_id.to_string(), score);
        }
        results.insert(query_id, query_results);
    }
    let results_json = serde_json::to_string_pretty(&results)?;
    let results_path = std::env::current_dir()?.join("results.json");
    let mut results_file = File::create(results_path)?;
    results_file.write_all(results_json.as_bytes())?;
    Ok(())
}
