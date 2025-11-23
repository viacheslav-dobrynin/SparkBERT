use anyhow::Result;
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use spark_bert::{
    api::{Config, SparkBert},
    util::device,
};
use std::sync::Arc;
use tokio::{net::TcpListener, sync::Mutex};

type SharedSparkBert = Arc<Mutex<SparkBert>>;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let config = Config {
        use_ram_index: true,
        device: device(false)?,
        index_n_neighbors: 8,
    };
    let spark_bert = Arc::new(Mutex::new(SparkBert::new(config)?));
    let app = Router::new()
        .route("/search", post(search))
        .route("/index", post(index_docs))
        .with_state(spark_bert);

    let listener = TcpListener::bind("0.0.0.0:8000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn search(
    State(spark_bert): State<SharedSparkBert>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    const DEFAULT_SEARCH_N_NEIGHBORS: usize = 3;
    const DEFAULT_TOP_K: usize = 10;

    let mut spark_bert = spark_bert.lock().await;
    let results = spark_bert
        .search(
            &payload.query,
            payload
                .search_n_neighbors
                .unwrap_or(DEFAULT_SEARCH_N_NEIGHBORS),
            payload.top_k.unwrap_or(DEFAULT_TOP_K),
        )
        .map_err(internal_error)?;

    let hits = results
        .into_iter()
        .map(|(doc_id, score)| SearchResult { doc_id, score })
        .collect();

    Ok(Json(SearchResponse { results: hits }))
}

fn internal_error(err: anyhow::Error) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    search_n_neighbors: Option<usize>,
    top_k: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Serialize)]
struct SearchResult {
    doc_id: u64,
    score: f64,
}

async fn index_docs(
    State(spark_bert): State<SharedSparkBert>,
    Json(payload): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, (StatusCode, String)> {
    let mut spark_bert = spark_bert.lock().await;
    let indexed = payload.docs.len();
    let docs = payload.docs.into_iter().map(|doc| (doc.doc_id, doc.text));
    spark_bert.index(docs, false).map_err(internal_error)?;
    Ok(Json(IndexResponse { indexed }))
}

#[derive(Deserialize)]
struct IndexRequest {
    docs: Vec<Doc>,
}

#[derive(Deserialize)]
struct Doc {
    doc_id: u64,
    text: String,
}

#[derive(Serialize)]
struct IndexResponse {
    indexed: usize,
}
