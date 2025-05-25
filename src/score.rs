use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;

pub fn calculate_max_sim(
    doc_embs: Vec<f32>,
    token_embs: Vec<f32>,
    device: &Device,
    d: usize,
) -> Result<Vec<f32>> {
    let doc_embs_count = doc_embs.len() / d;
    let doc_embs_tensor = Tensor::from_vec(doc_embs, (doc_embs_count, d), device)?;
    let token_embs_count = token_embs.len() / d;
    let token_embs_tensor = Tensor::from_vec(token_embs, (token_embs_count, d), device)?;
    let scores: Vec<f32> = doc_embs_tensor
        .matmul(&token_embs_tensor.t()?)?
        .max(0)?
        .to_vec1()?;
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_calculate_correct_max_sim_scores() {
        let doc_embs = vec![1f32, 2f32, 2f32, 1f32];
        let token_embs = vec![2f32, 4f32, 5f32, 1f32, 1f32, 1f32];
        let device = Device::Cpu;
        let d = 2;
        let scores = calculate_max_sim(doc_embs, token_embs, &device, d).unwrap();
        assert_eq!(scores, vec![10f32, 11f32, 3f32]);
    }
}
