use anyhow::Result;
use candle_core::Device;

use crate::{
    args::Args,
    embs::{convert_to_flatten_vec, Bert},
    inverted_index::InvertedIndex,
    score::calculate_max_sim,
    vector_vocabulary::VectorVocabulary,
};

pub struct SparkBert {
    vector_vocabulary: VectorVocabulary,
    inverted_index: InvertedIndex,
    bert: Bert,
    device: Device,
    index_n_neighbors: usize,
}
pub struct Config {
    pub use_ram_index: bool,
    pub device: Device,
    pub index_n_neighbors: usize,
}

impl SparkBert {
    pub fn new(config: Config) -> Result<Self> {
        let vector_vocabulary = VectorVocabulary::build()?;
        println!(
            "Vector vocabulary size: {}",
            vector_vocabulary.get_num_tokens()
        );
        let inverted_index = if config.use_ram_index {
            InvertedIndex::open_with_copy_from_disk_to_ram_directory()?
        } else {
            InvertedIndex::open()?
        };
        let args = Args {
            cpu: config.device.is_cpu(),
            tracing: false,
            model_id: Option::None,
            revision: Option::None,
            use_pth: false,
            normalize_embeddings: true,
            approximate_gelu: false,
        };
        let bert = Bert::build(args)?;
        Ok(Self {
            vector_vocabulary,
            inverted_index,
            bert,
            device: config.device,
            index_n_neighbors: config.index_n_neighbors,
        })
    }

    pub fn search(
        &mut self,
        query: &str,
        search_n_neighbors: usize,
        top_k: usize,
    ) -> Result<Vec<(u64, f64)>> {
        let query_embs = self.bert.calc_embs(vec![query], false)?;
        let flat_query_embs = convert_to_flatten_vec(&query_embs)?;
        let (tokens, _) =
            self.vector_vocabulary
                .find_tokens(&flat_query_embs, search_n_neighbors, false)?;
        let doc_id_score_pairs = self.inverted_index.search(None, tokens.as_slice(), top_k)?;
        Ok(doc_id_score_pairs)
    }

    pub fn index<I>(&mut self, docs: I) -> Result<()>
    where
        I: IntoIterator<Item = (u64, String)>,
    {
        let d = self.vector_vocabulary.get_embedding_dims() as usize;
        for (doc_id, text) in docs {
            let doc_embs = &self.bert.calc_embs(vec![text.as_str()], false)?;
            let flat_doc_embs = convert_to_flatten_vec(doc_embs)?;
            let (tokens, token_embs) =
                self.vector_vocabulary
                    .find_tokens(&flat_doc_embs, self.index_n_neighbors, true)?;
            let scores = calculate_max_sim(flat_doc_embs, token_embs.unwrap(), &self.device, d)?;
            self.inverted_index.index(
                doc_id,
                tokens.iter().map(|&it| it.to_owned()).collect(),
                scores,
            );
        }
        self.inverted_index.finalize()?;
        Ok(())
    }

    pub fn get_num_docs(&self) -> u64 {
        self.inverted_index.get_num_docs()
    }
}
