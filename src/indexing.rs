use std::time::Instant;

use anyhow::Result;
use candle_core::Device;

use crate::{
    api::{Config, SparkBert},
    dataset::Corpus,
    util::get_progress_bar,
};

pub fn build_spark_bert(
    corpus: &Corpus,
    index_n_neighbors: usize,
    device: Device,
) -> Result<SparkBert> {
    let inverted_index_building_start = Instant::now();
    let config = Config {
        use_ram_index: false,
        device,
        index_n_neighbors,
    };
    let mut spark_bert = SparkBert::new(config)?;
    let pb = get_progress_bar(corpus.len() as u64)?;
    let corpus_iter = pb.wrap_iter(corpus.iter().map(|p| {
        (
            p.0.parse().expect("Failed to parse string to u64"),
            p.1.as_text(),
        )
    }));
    spark_bert.index(corpus_iter, true)?;
    dbg!(inverted_index_building_start.elapsed());
    Ok(spark_bert)
}
