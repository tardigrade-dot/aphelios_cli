use anyhow::Result;
use aphelios_search::{
    build_index, extract_metadata, init_database, search_books, search_books_with_options,
    IndexConfig, SearchMode, SearchOptions,
};
use rusqlite::params;
use std::path::Path;
use tempfile::TempDir;

/// 创建测试用的临时数据库目录
fn create_test_env() -> (TempDir, IndexConfig) {
    let temp_dir = TempDir::new().expect("创建临时目录失败");
    let db_path = temp_dir.path().join("books.db");
    let semantic_path = temp_dir.path().join("embeddings.usearch");

    let config = IndexConfig {
        book_dir: temp_dir.path().to_string_lossy().to_string(),
        db_path: db_path.to_string_lossy().to_string(),
        semantic_index_path: semantic_path.to_string_lossy().to_string(),
        model_dir: "/tmp/nonexistent".to_string(),
    };

    (temp_dir, config)
}

/// 插入测试书籍数据
fn insert_test_books(config: &IndexConfig) {
    let conn = init_database(&config.db_path).expect("初始化数据库失败");

    let test_files = vec![
        // 文化、权力与国家 系列
        "文化、权力与国家：1900—1942年的华北农村 = Culture, Power, and the State (杜赞奇) (Z-Library).pdf",
        "文化、权力与国家：1900—1942年的华北农村 (杜赞奇) (Z-Library).epub",
        // 共产世界大历史 系列
        "共產世界大歷史  一個革命理想的形成與破滅 (5週年增訂新修版) (呂正理) (z-library.sk, 1lib.sk, z-lib.sk)_简体-no_img-横版.epub",
        "共產世界大歷史  一個革命理想的形成與破滅 (5週年增訂新修版) (呂正理) (z-library.sk, 1lib.sk, z-lib.sk).epub",
        // 其他书籍（用于测试噪音）
        "化学反应原理 (张三).epub",
        "世界历史教程 (李四).epub",
        "共产主义理想 (王五).epub",
    ];

    for (idx, filename) in test_files.iter().enumerate() {
        let path = format!(
            "/test/book_{}.{}",
            idx + 1,
            &filename[filename.rfind('.').unwrap_or(filename.len() - 4)..]
        );
        let file_type = filename.rsplit('.').next().unwrap_or("").to_lowercase();
        let size = 1000000u64;

        // 使用 extract_metadata 提取标题、作者和分词
        let (title, author, title_norm, author_norm, title_jieba, author_jieba) =
            extract_metadata(Path::new(filename));

        println!("插入: {} | Jieba: {}", title, title_jieba);

        conn.execute(
            "INSERT INTO books (title, author, title_norm, author_norm, title_bigram, author_bigram, file_path, file_type, file_size) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![title, author, title_norm, author_norm, title_jieba, author_jieba, path, file_type, size as i64],
        )
        .expect("插入书籍失败");

        // 更新 FTS 索引
        conn.execute(
            "INSERT INTO books_fts(rowid, title, author, title_norm, author_norm, title_bigram, author_bigram) VALUES (last_insert_rowid(), ?1, ?2, ?3, ?4, ?5, ?6)",
            params![title, author, title_norm, author_norm, title_jieba, author_jieba],
        )
        .expect("更新 FTS 失败");
    }

    println!("已插入 {} 本测试书籍", test_files.len());
}

#[test]
fn test_search_culture_power_state() {
    let (_temp_dir, config) = create_test_env();
    insert_test_books(&config);

    println!("\n========== 测试 1: 搜索 '文化、权力与国家' ==========");
    let options = SearchOptions {
        query: "文化、权力与国家".to_string(),
        limit: 20,
        mode: SearchMode::Keyword,
        normalize_chinese: true,
        config: config.clone(),
    };

    let result = search_books_with_options(&options).expect("搜索失败");
    for (i, book) in result.books.iter().enumerate() {
        println!("{:2}. [{}] {}", i + 1, book.file_type, book.title);
    }

    // 前两本应该是"文化、权力与国家"的 PDF 和 EPUB 版本
    assert!(result.books.len() >= 2, "应该至少找到 2 本相关书籍");
    assert!(
        result.books[0].title.contains("文化"),
        "排名第一的应该包含'文化'"
    );

    println!("\n========== 测试 2: 搜索 '文化权力与国家' (无标点) ==========");
    let options2 = SearchOptions {
        query: "文化权力与国家".to_string(),
        limit: 20,
        mode: SearchMode::Keyword,
        normalize_chinese: true,
        config: config.clone(),
    };

    let result2 = search_books_with_options(&options2).expect("搜索失败");
    for (i, book) in result2.books.iter().enumerate() {
        println!("{:2}. [{}] {}", i + 1, book.file_type, book.title);
    }

    assert!(result2.books.len() >= 2, "应该至少找到 2 本相关书籍");
}

#[test]
fn test_search_communism_world_history() {
    let (_temp_dir, config) = create_test_env();
    insert_test_books(&config);

    println!("\n========== 测试 3: 搜索 '共产世界大历史' ==========");
    let options = SearchOptions {
        query: "共产世界大历史".to_string(),
        limit: 20,
        mode: SearchMode::Keyword,
        normalize_chinese: true,
        config: config.clone(),
    };

    let result = search_books_with_options(&options).expect("搜索失败");
    for (i, book) in result.books.iter().enumerate() {
        println!("{:2}. [{}] {}", i + 1, book.file_type, book.title);
    }

    // 应该找到"共產世界大歷史"的两本书
    assert!(result.books.len() >= 2, "应该至少找到 2 本相关书籍");

    // 验证"共产世界大历史"相关的书在前 3 名中至少出现 2 次
    let top3_relevant = result
        .books
        .iter()
        .take(3)
        .filter(|b| b.title.contains("共產") || b.title.contains("共产"))
        .count();
    assert!(
        top3_relevant >= 2,
        "前 3 名中应该至少有 2 本'共产世界大历史'相关书籍"
    );
}

#[test]
fn test_jieba_tokenization_quality() {
    use aphelios_search::chinese_norm;

    // 测试 Jieba 分词质量
    let tokens1 = chinese_norm::tokenize_for_search("文化权力与国家");
    println!("\nJieba 分词 '文化权力与国家': {:?}", tokens1);
    assert!(tokens1.contains(&"文化".to_string()));
    assert!(
        !tokens1.contains(&"化权".to_string()),
        "不应该包含无意义的跨词组合'化权'"
    );

    let tokens2 = chinese_norm::tokenize_for_search("共产世界大历史");
    println!("Jieba 分词 '共产世界大历史': {:?}", tokens2);
    assert!(tokens2.contains(&"共产".to_string()) || tokens2.contains(&"共产主义".to_string()));
    assert!(tokens2.contains(&"世界".to_string()));
    assert!(
        !tokens2.contains(&"产世".to_string()),
        "不应该包含无意义的跨词组合'产世'"
    );
}

#[test]
fn test_folder_search() -> Result<()> {
    let index_config = IndexConfig::default();

    assert!(build_index(&index_config, None).is_ok());

    let search_keys = vec!["文化权力与国家", "共产世界大历史"];

    for s_key in search_keys {
        let search = search_books(&index_config, s_key, 5)?;
        println!("{}", s_key);
        for book in search.books {
            println!("  {}", book.title);
        }
    }
    Ok(())
}
