use std::sync::Arc;
use anyhow::Result;
use aphelios_core::traits::SearchResult;
use crate::controllers::AppContext;

pub struct SearchLogic {
    ctx: Arc<AppContext>,
}

impl SearchLogic {
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self { ctx }
    }

    pub fn ctx(&self) -> &Arc<AppContext> {
        &self.ctx
    }

    pub fn get_book_count(&self) -> Result<usize> {
        self.ctx.search_engine.get_book_count()
    }

    pub fn build_index(
        &self,
        on_complete: impl Fn(Result<usize>) + Send + 'static
    ) {
        let engine = self.ctx.search_engine.clone();

        std::thread::spawn(move || {
            let result = engine.build_index(None);
            on_complete(result);
        });
    }

    pub fn search_books(
        &self,
        query: String,
        limit: usize,
        on_complete: impl Fn(Result<SearchResult>) + Send + 'static
    ) {
        let engine = self.ctx.search_engine.clone();

        std::thread::spawn(move || {
            let result = engine.search_books(&query, limit);
            on_complete(result);
        });
    }
}
