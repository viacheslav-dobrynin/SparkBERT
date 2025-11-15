use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::faiss::{self, Index};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faiss::{self, error::Result as FaissResult, Idx};

    #[test]
    fn should_reconstruct_batch_of_embs() {
        let mock = MockIndex {
            vecs: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        };
        let labels = [Idx::new(0), Idx::new(1)];

        let embs = reconstruct_batch(&mock, &labels).unwrap();

        assert_eq!(embs, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    struct MockIndex {
        vecs: Vec<Vec<f32>>,
    }

    impl faiss::Index for MockIndex {
        fn d(&self) -> u32 {
            self.vecs[0].len() as u32
        }

        fn ntotal(&self) -> u64 {
            self.vecs.len() as u64
        }

        fn search(&mut self, _q: &[f32], _k: usize) -> FaissResult<faiss::index::SearchResult> {
            unimplemented!()
        }

        fn reconstruct(&self, idx: Idx, dest: &mut [f32]) -> FaissResult<()> {
            dest.copy_from_slice(&self.vecs[idx.get().unwrap() as usize]);
            Ok(())
        }
    }
}
