use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faiss::{read_index, Index};
use redis::Commands;
use spar_k_bert::{
    embs::calc_embs,
    run::{find_tokens, load_inverted_index},
};

fn bench_vector_search(c: &mut Criterion) {
    let mut vector_index = read_index("/home/slava/Developer/SparKBERT/hnsw.index").unwrap();
    let query = "some test query";

    let query_embs = calc_embs(vec![query], false).unwrap();
    let flat_embs = query_embs.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let search_n_neighbors = 3;
    let search_top_k = 10;
    let mut redis = redis::Client::open("redis://cache.home:16379")
        .unwrap()
        .get_connection()
        .unwrap();
    let faiss_idx_to_token: HashMap<String, String> = redis.hgetall("faiss_idx_to_token").unwrap();
    let inverted_index = load_inverted_index(&mut redis).unwrap();
    let tokens = find_tokens(
        &mut vector_index,
        &search_n_neighbors,
        &faiss_idx_to_token,
        query,
    )
    .unwrap();
    let tokens = tokens.as_slice();

    let mut group = c.benchmark_group("Vector Search");
    group.bench_function("HNSW", |b| {
        b.iter(|| {
            vector_index
                .search(black_box(&flat_embs), black_box(search_top_k))
                .unwrap()
        })
    });
    group.bench_function("SparKBERT", |b| {
        b.iter(|| {
            inverted_index
                .search(black_box(tokens), black_box(search_top_k))
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, bench_vector_search);
criterion_main!(benches);
