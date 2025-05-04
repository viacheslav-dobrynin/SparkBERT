use anyhow::{Error as E, Result};
use candle_core::Tensor;
use clap::Parser;
use tokenizers::PaddingParams;

pub fn calc_embs(sentences: Vec<&str>) -> Result<Tensor> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = crate::args::Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }
    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    println!("running inference on batch {:?}", token_ids.shape());
    let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
    println!("generated embeddings {:?}", embeddings.shape());
    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let embeddings = if args.normalize_embeddings {
        normalize_l2(&embeddings)?
    } else {
        embeddings
    };
    println!("Loaded and encoded {:?}", start.elapsed());
    println!("pooled embeddings {:?}", embeddings.shape());

    //let mut similarities = vec![];
    //for i in 0..n_sentences {
    //    let e_i = embeddings.get(i)?;
    //    for j in (i + 1)..n_sentences {
    //        let e_j = embeddings.get(j)?;
    //        let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
    //        let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
    //        let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
    //        let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
    //        similarities.push((cosine_similarity, i, j))
    //    }
    //}
    //similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    //for &(score, i, j) in similarities[..5].iter() {
    //    println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    //}
    Ok(embeddings)
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
