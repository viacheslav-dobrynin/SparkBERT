use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::embs::{calc_embs, convert_to_flatten_vec};
use anyhow::Result;
use candle_core::D;
use faiss::{Idx, Index};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

pub fn load_faiss_idx_to_token(json_path: &str) -> anyhow::Result<HashMap<String, String>> {
    let file = File::open(json_path)?;
    let reader = BufReader::new(file);
    let faiss_idx_to_token: HashMap<String, String> = serde_json::from_reader(reader)?;
    anyhow::Ok(faiss_idx_to_token)
}

pub fn reconstruct_batch<T>(index: &T, labels: &[faiss::Idx]) -> anyhow::Result<Vec<f32>>
where
    T: Index + Sync,
{
    let d = index.d() as usize;
    let batch = labels.len();
    let mut flat_embs = vec![0f32; batch * d];
    debug_assert_eq!(flat_embs.len(), labels.len() * d);
    flat_embs
        .par_chunks_mut(d)
        .enumerate()
        .try_for_each(|(i, chunk)| {
            let idx = labels[i];
            index.reconstruct(idx, chunk).map_err(anyhow::Error::from)
        })?;
    anyhow::Ok(flat_embs)
}

pub fn unique_labels(labels: &[Idx]) -> Vec<Idx> {
    let mut unique_ids: Vec<u64> = labels.iter().filter_map(|idx| idx.get()).collect();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    unique_ids.into_iter().map(Idx::new).collect()
}

// TODO: merge with fn from indexing.rs to avoid code duplicates
pub fn find_tokens<'a, T>(
    vector_index: &mut T,
    search_n_neighbors: &usize,
    faiss_idx_to_token: &'a HashMap<String, String>,
    query: &str,
) -> Result<Vec<&'a str>>
where
    T: Index + Sync,
{
    let query_embs = calc_embs(vec![query], false)?;
    let flat_embs = convert_to_flatten_vec(&query_embs)?;
    let faiss::index::SearchResult {
        distances: _,
        labels,
    } = vector_index.search(&flat_embs, *search_n_neighbors)?;
    let labels = unique_labels(&labels);
    let tokens: Vec<&str> = labels
        .iter()
        .map(|idx| {
            let idx = idx.get().unwrap().to_string();
            faiss_idx_to_token.get(&idx).map(String::as_str).unwrap()
        })
        .collect();
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faiss::error::Result as FaissResult;
    use faiss::{Idx, MetricType};

    #[test]
    fn should_reconstruct_batch_of_embs() {
        let mock = MockIndex {
            vecs: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        };
        let labels = [Idx::new(0), Idx::new(1)];

        let embs = reconstruct_batch(&mock, &labels).unwrap();

        assert_eq!(embs, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn should_return_unique_labels() {
        let labels = [
            Idx::new(2),
            Idx::new(1),
            Idx::new(2),
            Idx::new(3),
            Idx::new(1),
        ];

        let uniques = unique_labels(&labels);

        assert_eq!(uniques, vec![Idx::new(1), Idx::new(2), Idx::new(3)]);
    }

    struct MockIndex {
        vecs: Vec<Vec<f32>>,
    }

    impl faiss::Index for MockIndex {
        fn d(&self) -> u32 {
            self.vecs[0].len() as u32
        }

        fn reconstruct(&self, idx: Idx, dest: &mut [f32]) -> FaissResult<()> {
            dest.copy_from_slice(&self.vecs[idx.get().unwrap() as usize]);
            Ok(())
        }

        fn is_trained(&self) -> bool {
            todo!()
        }

        fn ntotal(&self) -> u64 {
            todo!()
        }

        fn metric_type(&self) -> MetricType {
            todo!()
        }

        fn add(&mut self, x: &[f32]) -> FaissResult<()> {
            let _ = x;
            todo!()
        }

        fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> FaissResult<()> {
            let _ = xids;
            let _ = x;
            todo!()
        }

        fn train(&mut self, x: &[f32]) -> FaissResult<()> {
            let _ = x;
            todo!()
        }

        fn assign(&mut self, q: &[f32], k: usize) -> FaissResult<faiss::index::AssignSearchResult> {
            let _ = k;
            let _ = q;
            todo!()
        }

        fn search(&mut self, q: &[f32], k: usize) -> FaissResult<faiss::index::SearchResult> {
            let _ = k;
            let _ = q;
            todo!()
        }

        fn range_search(
            &mut self,
            q: &[f32],
            radius: f32,
        ) -> FaissResult<faiss::index::RangeSearchResult> {
            let _ = radius;
            let _ = q;
            todo!()
        }

        fn reconstruct_n(
            &self,
            first_key: Idx,
            count: usize,
            output: &mut [f32],
        ) -> FaissResult<()> {
            let _ = output;
            let _ = count;
            let _ = first_key;
            todo!()
        }

        fn reset(&mut self) -> FaissResult<()> {
            todo!()
        }

        fn remove_ids(&mut self, sel: &faiss::selector::IdSelector) -> FaissResult<usize> {
            let _ = sel;
            todo!()
        }

        fn verbose(&self) -> bool {
            todo!()
        }

        fn set_verbose(&mut self, value: bool) {
            let _ = value;
            todo!()
        }
    }
}
