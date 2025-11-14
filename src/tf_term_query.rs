use tantivy::postings::{Postings, SegmentPostings};
use tantivy::query::{EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use tantivy::schema::IndexRecordOption;
use tantivy::{DocId, DocSet, Score, SegmentReader, Term};

// TF Scorer
struct TfScorer {
    postings: SegmentPostings,
}

impl DocSet for TfScorer {
    fn advance(&mut self) -> DocId {
        self.postings.advance();
        self.doc()
    }

    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        unimplemented!()
    }
}

impl Scorer for TfScorer {
    fn score(&mut self) -> Score {
        self.postings.term_freq() as Score
    }
}

#[derive(Debug, Clone)]
pub struct TfTermQuery {
    term: Term,
}

impl TfTermQuery {
    pub fn new(term: Term) -> Self {
        Self { term }
    }
}

struct TfWeight {
    term: Term,
}

impl Weight for TfWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> tantivy::Result<Box<dyn Scorer>> {
        let inv = reader.inverted_index(self.term.field())?;
        if let Some(info) = inv.get_term_info(&self.term)? {
            let postings = inv.read_postings_from_terminfo(&info, IndexRecordOption::WithFreqs)?;
            Ok(Box::new(TfScorer { postings }))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> tantivy::Result<Explanation> {
        unimplemented!()
    }
}

impl Query for TfTermQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> tantivy::Result<Box<dyn Weight>> {
        let weight = TfWeight {
            term: self.term.clone(),
        };
        tantivy::Result::Ok(Box::new(weight))
    }
}
