use anyhow::Result;
use aphelios_search::{build_index, IndexConfig};
use tracing::{error, info};

#[test]
fn test_build_index() -> Result<()> {
    // 使用默认配置（包含 metal feature 时会自动加速）
    let config = IndexConfig::default();

    info!(
        "Starting build index test with book_dir: {}",
        config.book_dir
    );
    info!("DB path: {}", config.db_path);
    info!("Semantic index path: {}", config.semantic_index_path);

    let r = build_index(
        &config,
        Some(&|progress| {
            info!("Index progress: {}%", progress);
        }),
    );

    match r {
        Ok(count) => {
            info!("Build index success, total books: {}", count);
            Ok(())
        }
        Err(e) => {
            error!("Build index failed: {}", e);
            Err(e)
        }
    }
}

#[test]
fn test_build_index_with_custom_dir() -> Result<()> {
    // 测试自定义目录
    let temp_dir = std::env::temp_dir().join("aphelios_search_test");
    std::fs::create_dir_all(&temp_dir).ok();

    let book_dir = temp_dir.to_string_lossy().to_string();
    let config = IndexConfig::from_book_dir(&book_dir);

    info!("Testing with temp dir: {}", book_dir);

    // 注意：如果 temp_dir 中没有书籍文件，这个测试会快速返回 0
    let r = build_index(
        &config,
        Some(&|progress| {
            info!("Index progress: {}%", progress);
        }),
    );

    match r {
        Ok(count) => {
            info!("Build index success, total books: {}", count);
            // 清理测试目录
            let _ = std::fs::remove_dir_all(&temp_dir);
            Ok(())
        }
        Err(e) => {
            error!("Build index failed: {}", e);
            let _ = std::fs::remove_dir_all(&temp_dir);
            Err(e)
        }
    }
}
