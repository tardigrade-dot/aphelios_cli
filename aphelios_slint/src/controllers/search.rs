use crate::controllers::AppContext;
use anyhow::Result;
use aphelios_core::traits::{IndexStatus, SearchMode, SearchResult};
use std::sync::Arc;

pub struct SearchLogic {
    ctx: Arc<AppContext>,
}

impl SearchLogic {
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self { ctx }
    }

    pub fn get_book_count(&self) -> Result<usize> {
        self.ctx.search_engine.get_book_count()
    }

    pub fn get_index_status(&self) -> Result<IndexStatus> {
        self.ctx.search_engine.get_index_status()
    }

    pub fn build_index(
        &self,
        book_dir: String,
        on_complete: impl Fn(Result<usize>) + Send + 'static,
    ) {
        let config = aphelios_search::IndexConfig::from_book_dir(&book_dir);

        std::thread::spawn(move || {
            let result = match aphelios_search::build_index(&config, None) {
                Ok(count) => Ok(count),
                Err(e) => Err(e),
            };
            on_complete(result);
        });
    }

    pub fn search_books_with_mode(
        &self,
        query: String,
        limit: usize,
        mode: SearchMode,
        on_complete: impl Fn(Result<SearchResult>) + Send + 'static,
    ) {
        let engine = self.ctx.search_engine.clone();

        std::thread::spawn(move || {
            let result = engine.search_books_with_mode(&query, limit, mode);
            on_complete(result);
        });
    }
}
