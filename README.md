# SparkBERT: **Spar**se representations using **k**-means and **BERT**

This repository contains a Rust implementation of the algorithm proposed in
[Efficient sparse retrieval through embedding-based inverted index construction](https://ntv.ifmo.ru/en/article/23315/effektivnyy_razrezhennyy_poisk_s_pomoschyu_postroeniya_invertirovannogo_indeksa_na_osnove_embeddingov%C2%A0.htm).

#### TL;DR

Vector search underpins many modern search engines thanks to its strong quality.
It is also widely used when building LLM agents for Retrieval-Augmented Generation (RAG).
The most commonly used algorithm for vector indexing is
Hierarchical Navigable Small World (HNSW) (see Table 3 in [this paper](https://arxiv.org/abs/2310.11703)).
Its main drawback is high memory consumption.
Moreover, Google DeepMind has shown that for a fixed vector dimensionality, beyond a certain number of documents
retrieving all relevant documents becomes mathematically impossible (see [this paper](https://arxiv.org/abs/2508.21038)).

This work proposes a hybrid retrieval algorithm that combines the quality of deep neural networks such as BERT with the resource efficiency of an inverted index.
To achieve this, it uses a so‑called vector vocabulary.
Its construction is based on the idea that each word can appear in different usage contexts, which can be captured in the embedding of that word produced by BERT in a given context.
By collecting as many such context-aware embeddings as possible, we can cluster them and obtain a set of semantic meanings for the word.
The centroids representing these semantic meanings become the elements of the vector vocabulary.

The resulting vector vocabulary is used both during indexing and during search.
It is important to note that for relevance scoring we use the MaxSim function proposed by the authors of [ColBERT](https://github.com/stanford-futuredata/ColBERT), which improves the quality of the scoring.

As a result, the algorithm delivers search quality comparable to HNSW while requiring less memory.

