use anyhow::{Context, Result};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

/// ---------- типы данных ----------
#[derive(Debug, Deserialize)]
pub struct CorpusDoc {
    pub title: Option<String>,
    pub text: String,
    #[serde(flatten)]
    pub meta: HashMap<String, serde_json::Value>,
}
#[derive(Debug, Deserialize)]
pub struct Query {
    pub text: String,
}

pub type Corpus = HashMap<String, CorpusDoc>;
pub type Queries = HashMap<String, Query>;
pub type Qrels = HashMap<String, HashMap<String, i32>>;

/// базовая директория с уже-скачанным датасетом
const DATA_DIR: &str = "datasets/scifact";

/// ---------- утилиты чтения ----------
fn read_jsonl<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<HashMap<String, T>> {
    let file = BufReader::new(File::open(&path)?);
    let mut map = HashMap::new();

    for line in file.lines() {
        let line = line?;
        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&line)?;
        let id = obj
            .remove("_id")
            .context("_id field missing")?
            .as_str()
            .unwrap()
            .to_owned();

        let val: T = serde_json::from_value(serde_json::Value::Object(obj))?;
        map.insert(id, val);
    }
    Ok(map)
}

fn read_qrels(path: impl AsRef<Path>) -> Result<Qrels> {
    let file = BufReader::new(File::open(&path)?);
    let mut map: Qrels = HashMap::new();

    for line in file.lines() {
        let line = line?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 3 || parts[0] == "query-id" {
            println!("First line or wrong len: {}", parts.len());
            continue;
        }
        let (q, d, rel) = (parts[0], parts[1], parts[2].parse::<i32>()?);
        map.entry(q.to_owned())
            .or_default()
            .insert(d.to_owned(), rel);
    }
    Ok(map)
}

/// ---------- публичная функция ----------
pub fn load_scifact(split: &str) -> Result<(Corpus, Queries, Qrels)> {
    let base = PathBuf::from(DATA_DIR);

    let corpus = read_jsonl(base.join("corpus.jsonl")).with_context(|| "reading corpus.jsonl")?;

    let mut queries =
        read_jsonl(base.join("queries.jsonl")).with_context(|| "reading queries.jsonl")?;

    let qrels = read_qrels(base.join(format!("qrels/{split}.tsv")))
        .with_context(|| format!("reading qrels/{split}.tsv"))?;

    queries.retain(|qid, _| qrels.contains_key(qid));

    Ok((corpus, queries, qrels))
}
