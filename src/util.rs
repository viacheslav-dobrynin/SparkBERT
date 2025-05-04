use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};
use faiss::Index;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn reconstruct_batch(
    index: &faiss::index::IndexImpl,
    labels: &[faiss::Idx],
) -> anyhow::Result<Vec<f32>> {
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
    //labels
    //    .par_iter()
    //    .map(|&idx| {
    //        let mut emb = vec![0.0f32; d];
    //        index.reconstruct(idx, &mut emb)?;
    //        anyhow::Ok(emb)
    //    })
    //    .collect::<anyhow::Result<Vec<_>>>()
    anyhow::Ok(flat_embs)
}
