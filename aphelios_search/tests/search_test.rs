use aphelios_search::{BookSearcher, SearchOptions, BOOK_EXTENSIONS};
use std::fs;
use std::path::PathBuf;

/// 创建一个独立的一次性临时目录
fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aphelios_search_test_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// 用简体搜索繁体文件名
#[test]
fn simplified_query_matches_traditional_file_name() {
    let dir = temp_dir("trad");
    fs::write(dir.join("三體.pdf"), "x").unwrap();
    fs::write(dir.join("深度学习入门.epub"), "x").unwrap();
    fs::write(dir.join("机器学习.txt"), "x").unwrap();

    let searcher = BookSearcher::new();
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "三体".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1, "应命中「三體.pdf」: {hits:?}");
    assert_eq!(hits[0].file_name, "三體.pdf");

    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "深度学习".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_name, "深度学习入门.epub");

    let _ = fs::remove_dir_all(&dir);
}

/// 用繁体搜索简体文件名
#[test]
fn traditional_query_matches_simplified_file_name() {
    let dir = temp_dir("simp");
    fs::write(dir.join("机器学习.txt"), "x").unwrap();

    let searcher = BookSearcher::new();
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "機器學習".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1, "应命中「机器学习.txt」: {hits:?}");
    assert_eq!(hits[0].file_name, "机器学习.txt");

    let _ = fs::remove_dir_all(&dir);
}

/// 按文件类型过滤
#[test]
fn filter_by_extension() {
    let dir = temp_dir("ext");
    fs::write(dir.join("a.pdf"), "x").unwrap();
    fs::write(dir.join("b.epub"), "x").unwrap();
    fs::write(dir.join("c.txt"), "x").unwrap();

    let searcher = BookSearcher::new();
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            extensions: Some(vec!["pdf".into()]),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_name, "a.pdf");

    // 带点号 / 大写扩展名也能匹配
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            extensions: Some(vec![".EPUB".into()]),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_name, "b.epub");

    let _ = fs::remove_dir_all(&dir);
}

/// 默认类型过滤: 只收 BOOK_EXTENSIONS 里的书籍文件
#[test]
fn default_extensions_only() {
    let dir = temp_dir("default_ext");
    fs::write(dir.join("book.pdf"), "x").unwrap();
    fs::write(dir.join("notes.docx"), "x").unwrap();
    fs::write(dir.join("photo.jpg"), "x").unwrap(); // 非书籍类型, 不应出现

    let searcher = BookSearcher::new();
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "".into(),
            ..Default::default()
        })
        .unwrap();
    let names: Vec<&str> = hits
        .iter()
        .map(|h| h.file_name.as_str())
        .collect();
    assert!(names.contains(&"book.pdf"));
    assert!(names.contains(&"notes.docx"));
    assert!(!names.contains(&"photo.jpg"));
    assert!(BOOK_EXTENSIONS.contains(&"pdf"));
    assert!(BOOK_EXTENSIONS.contains(&"docx"));

    let _ = fs::remove_dir_all(&dir);
}

/// 递归子目录 + 多关键字 AND 匹配
#[test]
fn recursive_and_multi_term() {
    let dir = temp_dir("recursive");
    fs::create_dir_all(dir.join("作者/邱锡鹏")).unwrap();
    fs::write(dir.join("作者/邱锡鹏/神经网络与深度学习.pdf"), "x").unwrap();
    fs::write(dir.join("深度学习框架入门.epub"), "x").unwrap();

    let searcher = BookSearcher::new();
    // 多关键字: 两个词都出现才命中
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "神经网络 深度学习".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_name, "神经网络与深度学习.pdf");

    // 递归: 子目录里的文件能找到; 同时目录名也能参与匹配
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "邱锡鹏".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_name, "神经网络与深度学习.pdf");

    let _ = fs::remove_dir_all(&dir);
}

/// 分隔符不影响匹配: 查询「深度学习 入门」能命中「深度学习-入门」
#[test]
fn separators_are_ignored() {
    let dir = temp_dir("sep");
    fs::write(dir.join("深度学习-入门.pdf"), "x").unwrap();

    let searcher = BookSearcher::new();
    let hits = searcher
        .search(&SearchOptions {
            dir: dir.clone(),
            query: "深度学习 入门".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hits.len(), 1, "应命中「深度学习-入门.pdf」: {hits:?}");

    let _ = fs::remove_dir_all(&dir);
}

/// 不存在的目录应报错
#[test]
fn missing_dir_returns_error() {
    let searcher = BookSearcher::new();
    let result = searcher.search(&SearchOptions {
        dir: PathBuf::from("/definitely/not/exist/xyz"),
        ..Default::default()
    });
    assert!(result.is_err());
}
