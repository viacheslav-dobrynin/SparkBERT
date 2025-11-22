use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_transformers::models::bert::BertModel;
use tokenizers::{PaddingParams, Tokenizer};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

use crate::args::Args;

pub struct Bert {
    model: BertModel,
    tokenizer: Tokenizer,
    args: Args,
}

impl Bert {
    pub fn build(args: Args) -> Result<Self> {
        let (model, tokenizer) = args.build_model_and_tokenizer()?;
        Ok(Self {
            model,
            tokenizer,
            args,
        })
    }

    pub fn calc_embs(&mut self, sentences: Vec<&str>, apply_pooling: bool) -> Result<Tensor> {
        let _guard = if self.args.tracing {
            println!("tracing...");
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };
        let start = std::time::Instant::now();

        let device = &self.model.device;

        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self
            .tokenizer
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
        //println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        //println!("generated embeddings {:?}", embeddings.shape());
        let embeddings = if apply_pooling {
            // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
            (embeddings.sum(1)? / (n_tokens as f64))?
        } else {
            embeddings
        };
        let embeddings = if apply_pooling && self.args.normalize_embeddings {
            normalize_l2(&embeddings)?
        } else {
            embeddings
        };
        //println!("Loaded and encoded {:?}", start.elapsed());
        //println!("pooled embeddings {:?}", embeddings.shape());

        Ok(embeddings)
    }
}

// TODO: adapt to 3D vector
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn convert_to_flatten_vec(embs: &Tensor) -> Result<Vec<f32>> {
    Ok(embs.flatten_all()?.to_vec1::<f32>()?)
}
