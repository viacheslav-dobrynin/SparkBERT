use std::{path::PathBuf, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faiss::{read_index, Index};
use spark_bert::{
    api::{Config, SparkBert},
    args::Args,
    embs::{convert_to_flatten_vec, Bert},
    util::device,
};

fn bench_vector_search(c: &mut Criterion) {
    // Setup common
    let search_top_k = 1000;
    let queries = vec![
        "some test query",
        "Bariatric surgery has a positive impact on mental health.",
        "All hematopoietic stem cells segregate their chromosomes randomly.",
    ];
    let device = device(false).unwrap();
    let args = Args {
        cpu: device.is_cpu(),
        tracing: false,
        model_id: Option::None,
        revision: Option::None,
        use_pth: false,
        normalize_embeddings: true,
        approximate_gelu: false,
    };
    let mut bert = Bert::build(args).unwrap();
    // Setup HNSW
    let hnsw_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/scifact.hnsw.faiss")
        .into_os_string()
        .into_string()
        .unwrap();
    let mut scifact_vector_index = read_index(hnsw_path).unwrap();
    println!(
        "Scifact vector index size: {}",
        scifact_vector_index.ntotal()
    );

    // Setup SparKBERT
    let config = Config {
        use_ram_index: true,
        device: device.to_owned(),
        index_n_neighbors: 8,
    };
    let mut spark_bert = SparkBert::new(config).unwrap();
    let search_n_neighbors = 3;
    println!(
        "SparkBERT index size: {}",
        spark_bert.get_num_docs().unwrap()
    );

    // Setup benchmark
    let mut group = c.benchmark_group("Vector Search");
    group.warm_up_time(Duration::from_secs(10));

    for query in &queries {
        let bench_name = &query.split_once(" ").unwrap().0;

        group.bench_function(format!("HNSW/{}", bench_name), |b| {
            b.iter(|| {
                let query_embs = bert.calc_embs(vec![black_box(query)], true).unwrap();
                let flat_embs = convert_to_flatten_vec(black_box(&query_embs)).unwrap();
                scifact_vector_index
                    .search(black_box(&flat_embs), black_box(search_top_k))
                    .unwrap()
            })
        });
        group.bench_function(format!("SparKBERT/{}", bench_name), |b| {
            b.iter(|| {
                spark_bert
                    .search(
                        black_box(query),
                        black_box(search_n_neighbors),
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
