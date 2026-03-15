use axum::{extract::Path, response::Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web-dist"]
#[exclude = "*.wasm"]
struct Assets;

async fn handler(Path(path): Path<String>) -> Response {
    let path = if path.is_empty() { "index.html" } else { &path };

    if let Some(content) = Assets::get(path) {
        Response::builder().body(content.data.into()).unwrap()
    } else {
        let index = Assets::get("index.html").unwrap();

        Response::builder().body(index.data.into()).unwrap()
    }
}
