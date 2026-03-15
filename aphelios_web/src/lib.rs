use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use rust_embed::RustEmbed;
use serde::Serialize;
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// 嵌入 web-dist 目录的静态文件
#[derive(RustEmbed)]
#[folder = "web-dist"]
#[exclude = "*.wasm"]
struct WebAssets;

#[derive(Clone)]
pub struct AppState {
    pub version: String,
}

#[derive(OpenApi)]
#[openapi(
    paths(health_handler, version_handler),
    components(schemas(HealthResponse, VersionResponse)),
    tags(
        (name = "api", description = "API endpoints"),
        (name = "web", description = "Web frontend")
    ),
    info(
        title = "Aphelios Web API",
        version = "0.1.0",
        description = "Aphelios CLI Web Interface API"
    )
)]
pub struct ApiDoc;

#[derive(Serialize, utoipa::ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub code: u16,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct VersionResponse {
    pub version: String,
    pub status: String,
    pub name: String,
}

/// 健康检查接口
#[utoipa::path(
    get,
    path = "/api/health",
    tag = "api",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
    )
)]
pub async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        axum::Json(HealthResponse {
            status: "healthy".to_string(),
            code: 200,
        }),
    )
}

/// 版本信息接口
#[utoipa::path(
    get,
    path = "/api/version",
    tag = "api",
    responses(
        (status = 200, description = "Version information", body = VersionResponse),
    )
)]
pub async fn version_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        axum::Json(VersionResponse {
            version: state.version.clone(),
            status: "running".to_string(),
            name: "aphelios_web".to_string(),
        }),
    )
}

/// 根路径处理器
pub async fn root_handler() -> impl IntoResponse {
    serve_static_file("index.html").await
}

/// 静态文件处理器
pub async fn static_handler(path: axum::extract::Path<String>) -> impl IntoResponse {
    let path = path.trim_start_matches('/');
    serve_static_file(path).await
}

async fn serve_static_file(path: &str) -> impl IntoResponse {
    let path = if path.is_empty() { "index.html" } else { path };

    match WebAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, mime.as_ref())],
                content.data,
            )
                .into_response()
        }
        None => {
            // 如果是前端路由，返回 index.html
            if !path.contains('.') {
                if let Some(content) = WebAssets::get("index.html") {
                    let mime = mime_guess::from_path("index.html").first_or_octet_stream();
                    (
                        StatusCode::OK,
                        [(axum::http::header::CONTENT_TYPE, mime.as_ref())],
                        content.data,
                    )
                        .into_response()
                } else {
                    (StatusCode::NOT_FOUND, "404 Not Found").into_response()
                }
            } else {
                (StatusCode::NOT_FOUND, "404 Not Found").into_response()
            }
        }
    }
}

/// 启动 Web 服务器的公共函数（可被其他 crate 调用）
pub async fn run_web_server(addr: &str) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        version: env!("CARGO_PKG_VERSION").to_string(),
    });

    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let display_addr = addr.replace("0.0.0.0", "127.0.0.1");
    println!();
    println!("🚀 Aphelios Web Server");
    println!("   📡 Listening on: http://{}", display_addr);
    println!("   📄 Frontend:     http://{}/", display_addr);
    println!("   📚 Swagger UI:   http://{}/swagger-ui", display_addr);
    println!(
        "   📋 OpenAPI JSON: http://{}/api-docs/openapi.json",
        display_addr
    );
    println!("   ❤️  Health:       http://{}/api/health", display_addr);
    println!("   📦 Version:      http://{}/api/version", display_addr);
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}

/// 创建路由器（方便自定义配置）
pub fn create_router(state: Arc<AppState>) -> Router {
    // 使用 utoipa-swagger-ui 的 axum 集成
    let swagger_router =
        SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi());

    Router::new()
        // API 路由
        .route("/api/health", get(health_handler))
        .route("/api/version", get(version_handler))
        // 静态文件服务
        .route("/", get(root_handler))
        .route("/*path", get(static_handler))
        // 合并 Swagger UI (放在最后，避免冲突)
        .merge(swagger_router)
        .with_state(state)
}
