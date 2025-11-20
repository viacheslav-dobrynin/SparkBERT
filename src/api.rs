use anyhow::Result;

use crate::{
    embs::{calc_embs, convert_to_flatten_vec},
    inverted_index::InvertedIndex,
    vector_vocabulary::VectorVocabulary,
};

pub struct SparkBert {
    vector_vocabulary: VectorVocabulary,
    inverted_index: InvertedIndex,
}
pub struct Config {
    pub use_ram_index: bool,
}

impl SparkBert {
    pub fn new(config: Config) -> Result<Self> {
        let vector_vocabulary = VectorVocabulary::build()?;
        let inverted_index = if config.use_ram_index {
            InvertedIndex::open_with_copy_from_disk_to_ram_directory()?
        } else {
            InvertedIndex::open()?
        };
        Ok(Self {
            vector_vocabulary,
            inverted_index,
        })
    }

    pub fn search(
        &mut self,
        query: &str,
        search_n_neighbors: usize,
        top_k: usize,
    ) -> Result<Vec<(u64, f64)>> {
        let query_embs = calc_embs(vec![query], false)?;
        let flat_query_embs = convert_to_flatten_vec(&query_embs)?;
        let (tokens, _) =
            self.vector_vocabulary
                .find_tokens(&flat_query_embs, search_n_neighbors, false)?;
        let doc_id_score_pairs = self.inverted_index.search(None, tokens.as_slice(), top_k)?;
        Ok(doc_id_score_pairs)
    }

    pub fn index<I>(&self, docs: I) -> Result<()>
    where
        I: IntoIterator<Item = (u64, String)>,
    {
        for (doc_id, text) in docs {}
        Ok(())
    }
}
