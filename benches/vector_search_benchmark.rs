use std::{collections::HashMap, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faiss::{read_index, Index};
use spark_bert::{
    embs::{calc_embs, convert_to_flatten_vec},
    inverted_index::InvertedIndex,
    vector_index::VectorVocabulary,
};

fn bench_vector_search(c: &mut Criterion) {
    // Setup common
    let search_top_k = 1000;
    let queries = vec![
        "some test query",
        "Bariatric surgery has a positive impact on mental health.",
        "All hematopoietic stem cells segregate their chromosomes randomly.",
    ];
    // Setup HNSW
    let mut scifact_vector_index =
        read_index("/home/slava/Developer/SparKBERT/scifact.hnsw.faiss").unwrap();
    println!(
        "Scifact vector index size: {}",
        scifact_vector_index.ntotal()
    );

    // Setup SparKBERT
    let mut vector_vocabulary = VectorVocabulary::build().unwrap();
    println!(
        "Vector vocabulary size: {}",
        vector_vocabulary.get_num_tokens()
    );
    let search_n_neighbors = 3;
    let inverted_index = InvertedIndex::open_with_copy_from_disk_to_ram_directory().unwrap();
    println!(
        "Inverted index size: {}",
        inverted_index.get_num_docs().unwrap()
    );
    let searcher = inverted_index.reader.as_ref().unwrap().searcher();

    // Setup benchmark
    let mut group = c.benchmark_group("Vector Search");
    group.warm_up_time(Duration::from_secs(10));
    for query in &queries {
        let query_embs = calc_embs(vec![query], true).unwrap();
        let flat_embs = convert_to_flatten_vec(&query_embs).unwrap();
        let (tokens, _) = vector_vocabulary
            .find_tokens(&flat_embs, search_n_neighbors, false)
            .unwrap();
        let tokens = tokens.as_slice();

        let bench_name = &query.split_once(" ").unwrap().0;

        group.bench_function(format!("HNSW/{}", bench_name), |b| {
            b.iter(|| {
                scifact_vector_index
                    .search(black_box(&flat_embs), black_box(search_top_k))
                    .unwrap()
            })
        });
        group.bench_function(format!("SparKBERT/{}", bench_name), |b| {
            b.iter(|| {
                inverted_index
                    .search(
                        black_box(Some(&searcher)),
                        black_box(tokens),
                        black_box(search_top_k),
                    )
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vector_search);
criterion_main!(benches);
