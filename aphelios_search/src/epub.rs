//! EPUB 解析: 把 EPUB 解析为真实的章节结构(Markdown), 并提取图片。
//!
//! 基于 `rbook`(维护活跃、TOC/资源 API 完善), 不依赖任何 UI 代码。

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use rbook::Epub;

/// 一本书的章节(按目录结构聚合的「真实章节」, 而非 spine 文件数)
#[derive(Clone, Debug)]
pub struct EpubChapter {
    /// 章节序号(0 起)
    pub index: usize,
    /// 章节标题(来自目录)
    pub title: String,
    /// Markdown 格式的内容(该章节全部 spine 页面拼接)
    pub markdown: String,
}

/// 目录条目(层级结构)
#[derive(Clone, Debug)]
pub struct EpubTocEntry {
    pub label: String,
    /// 层级(0 起)
    pub depth: usize,
    /// 对应的章节序号(该目录项有正文页时为 Some)
    pub chapter: Option<usize>,
    pub children: Vec<EpubTocEntry>,
}

/// 解析后的 EPUB 书籍
#[derive(Clone, Debug)]
pub struct EpubBook {
    pub path: PathBuf,
    pub title: String,
    pub author: String,
    pub chapters: Vec<EpubChapter>,
    pub toc: Vec<EpubTocEntry>,
    /// 提取出的图片: 原 src(相对路径) -> 临时文件绝对路径
    pub images: HashMap<String, PathBuf>,
    /// 图片临时目录
    pub temp_dir: PathBuf,
}

impl EpubBook {
    /// 打开并解析 EPUB
    pub fn open(path: &Path) -> Result<Self> {
        let epub = Epub::open(path)
            .with_context(|| format!("无法打开 EPUB 文件: {}", path.display()))?;

        let title = epub
            .metadata()
            .titles()
            .next()
            .map(|t| t.value().to_string())
            .or_else(|| path.file_stem().map(|s| s.to_string_lossy().to_string()))
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| "未命名书籍".to_string());

        let author = epub
            .metadata()
            .creators()
            .next()
            .map(|c| c.value().to_string())
            .filter(|a| !a.trim().is_empty())
            .unwrap_or_default();

        // 1. 读取 spine 全部页面(保留空页面占位, 保证目录的 spine 索引始终有效)
        let mut pages: Vec<Option<String>> = Vec::new();
        for entry in epub.spine().iter() {
            let md = entry.manifest_entry().and_then(|me| me.read_str().ok()).map(|html| {
                let md = xhtml_to_markdown(&html).unwrap_or_else(|_| strip_tags(&html));
                md.trim().to_string()
            });
            pages.push(md.filter(|m| !m.is_empty()));
        }
        if pages.iter().all(|p| p.is_none()) {
            anyhow::bail!("没有可读取的正文内容");
        }

        // 2. 收集目录(spine 中的位置由 manifest id 关联)
        let toc_tree = collect_toc(&epub);

        // 3. 按目录把 spine 页面聚合成真实章节
        let (chapters, chapter_of_spine) = group_chapters(&pages, &toc_tree);

        // 4. 关联目录项 -> 章节
        let toc = toc_tree
            .into_iter()
            .map(|n| attach_chapters(n, &chapter_of_spine))
            .collect();

        // 5. 提取图片到临时目录
        let temp_dir = std::env::temp_dir().join(format!("aphelios_epub_{:016x}", path_hash(path)));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).ok();

        let mut images: HashMap<String, PathBuf> = HashMap::new();
        let chapters = chapters
            .into_iter()
            .map(|mut ch| {
                extract_images(&epub, &mut ch.markdown, &temp_dir, &mut images);
                ch
            })
            .collect::<Vec<_>>();

        Ok(Self {
            path: path.to_path_buf(),
            title,
            author,
            chapters,
            toc,
            images,
            temp_dir,
        })
    }

    pub fn chapter_count(&self) -> usize {
        self.chapters.len()
    }

    pub fn chapter(&self, index: usize) -> Option<&EpubChapter> {
        self.chapters.get(index)
    }
}

/// 目录原始节点
struct TocNode {
    label: String,
    depth: usize,
    /// 在 spine 中的位置
    spine: Option<usize>,
    children: Vec<TocNode>,
}

fn spine_index_of(epub: &Epub, manifest_id: &str) -> Option<usize> {
    epub.spine()
        .iter()
        .position(|entry| {
            entry
                .manifest_entry()
                .map(|me| me.id() == manifest_id)
                .unwrap_or(false)
        })
}

fn collect_toc(epub: &Epub) -> Vec<TocNode> {
    let Some(root) = epub.toc().contents() else {
        return Vec::new();
    };
    root.iter().map(|e| collect_toc_node(e, epub)).collect()
}

fn collect_toc_node(entry: rbook::epub::toc::EpubTocEntry<'_>, epub: &Epub) -> TocNode {
    let spine = entry
        .manifest_entry()
        .and_then(|me| spine_index_of(epub, me.id()));
    let children = entry.iter().map(|c| collect_toc_node(c, epub)).collect();
    TocNode {
        label: entry.label().to_string(),
        depth: entry.depth(),
        spine,
        children,
    }
}

/// 单个章节的最大字符数(超过则用目录的嵌套条目拆分为小节,
/// 仍超过则按 spine 页面拆分, 避免整卷/大章塞进一个章节导致渲染卡顿)
const MAX_CHAPTER_CHARS: usize = 40_000;

/// 按目录把 spine 页面分组为真实章节
///
/// `pages` 与 spine 一一对应(空页面为 None), 因此目录的 spine 索引可以直接使用。
/// 返回 (章节列表, spine 位置 -> 章节序号)
fn group_chapters(pages: &[Option<String>], toc: &[TocNode]) -> (Vec<EpubChapter>, Vec<usize>) {
    // 顶层目录项(有 spine 位置)构成章节边界
    let mut boundaries: Vec<(usize, String)> = toc
        .iter()
        .filter_map(|n| n.spine.map(|s| (s, n.label.clone())))
        .filter(|(s, _)| *s < pages.len())
        .collect();
    boundaries.sort_by_key(|(s, _)| *s);

    // 全部目录项的 spine 标记(任意层级), 用于拆分超大章节
    let mut all_marks: Vec<(usize, String)> = Vec::new();
    for node in toc {
        collect_marks(node, &mut all_marks);
    }
    all_marks.sort_by_key(|(s, _)| *s);
    all_marks.dedup_by_key(|(s, _)| *s);

    let mut chapter_specs: Vec<(String, usize, usize)> = Vec::new(); // (title, start, end)
    if boundaries.is_empty() {
        // 无目录: 每个 spine 页面一章
        for (i, page) in pages.iter().enumerate() {
            let title = page
                .as_ref()
                .and_then(|md| first_heading(md))
                .unwrap_or_else(|| format!("第 {} 章", chapter_specs.len() + 1));
            chapter_specs.push((title, i, i + 1));
        }
    } else {
        let first = boundaries[0].0;
        if first > 0 {
            // 目录前的页面(封面/卷首等)归入第一章开头
            chapter_specs.push(("卷首".to_string(), 0, first));
        }
        for (k, &(start, ref label)) in boundaries.iter().enumerate() {
            let end = boundaries.get(k + 1).map(|&(e, _)| e).unwrap_or(pages.len());
            let title = if label.trim().is_empty() {
                pages[start]
                    .as_ref()
                    .and_then(|md| first_heading(md))
                    .unwrap_or_else(|| format!("第 {} 章", chapter_specs.len() + 1))
            } else {
                label.clone()
            };
            chapter_specs.push((title, start, end));
        }
    }

    // 超大章节按嵌套目录条目拆分
    let mut subdivided: Vec<(String, usize, usize)> = Vec::new();
    for (title, start, end) in chapter_specs {
        let size: usize = pages[start..end.min(pages.len())]
            .iter()
            .filter_map(|p| p.as_deref())
            .map(str::len)
            .sum();
        if size <= MAX_CHAPTER_CHARS {
            subdivided.push((title, start, end));
            continue;
        }
        // 该章节范围内的嵌套目录标记(不含起点自身)
        let subs: Vec<(usize, String)> = all_marks
            .iter()
            .filter(|(s, _)| *s > start && *s < end)
            .cloned()
            .collect();
        if subs.is_empty() {
            // 目录里没有嵌套条目可拆分: 退化为按 spine 页面拆分
            for page in start..end.min(pages.len()) {
                subdivided.push((title.clone(), page, page + 1));
            }
            continue;
        }
        subdivided.push((title.clone(), start, subs[0].0));
        for (k, &(ms, ref mlabel)) in subs.iter().enumerate() {
            let mend = subs.get(k + 1).map(|&(e, _)| e).unwrap_or(end);
            let mt = if mlabel.trim().is_empty() {
                title.clone()
            } else {
                mlabel.clone()
            };
            subdivided.push((mt, ms, mend));
        }
    }

    // 最终兜底: 仍超过上限的章节(嵌套目录条目太少/太大)按 spine 页面拆分
    let mut final_specs: Vec<(String, usize, usize)> = Vec::new();
    for (title, start, end) in subdivided {
        let size: usize = pages[start..end.min(pages.len())]
            .iter()
            .filter_map(|p| p.as_deref())
            .map(str::len)
            .sum();
        if size <= MAX_CHAPTER_CHARS {
            final_specs.push((title, start, end));
        } else {
            for page in start..end.min(pages.len()) {
                final_specs.push((title.clone(), page, page + 1));
            }
        }
    }

    // 组装章节: 空章节直接跳过(封面/导航等无正文页), 索引保持连续
    let mut chapter_of_spine = vec![0usize; pages.len()];
    let mut chapters: Vec<EpubChapter> = Vec::new();
    for (title, start, end) in final_specs {
        let end = end.min(pages.len());
        let markdown = pages[start..end]
            .iter()
            .filter_map(|p| p.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n");
        if markdown.trim().is_empty() {
            continue;
        }
        // 单个 spine 页面仍可能超大(如整章塞在一个 xhtml 里), 按段落拆分为小节
        let pieces = split_markdown(&markdown, MAX_CHAPTER_CHARS);
        let total = pieces.len();
        for (k, piece) in pieces.into_iter().enumerate() {
            let piece_title = if total > 1 {
                format!("{title}（{}/{}）", k + 1, total)
            } else {
                title.clone()
            };
            let index = chapters.len();
            for s in start..end {
                chapter_of_spine[s] = index;
            }
            chapters.push(EpubChapter {
                index,
                title: piece_title,
                markdown: piece,
            });
        }
    }

    (chapters, chapter_of_spine)
}


/// 把超大的 markdown 拆分为若干块(每块不超过 max)
///
/// 优先在段落边界拆分; 单段过长时按换行拆分; 仍过长时按字符硬切。
fn split_markdown(markdown: &str, max: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    let push_current = |chunks: &mut Vec<String>, current: &mut String| {
        if !current.is_empty() {
            chunks.push(std::mem::take(current));
        }
    };

    for block in markdown.split("\n\n") {
        if block.len() > max {
            push_current(&mut chunks, &mut current);
            // 单个块(如超长段落/代码块)太大: 按行拆
            let mut piece = String::new();
            for line in block.split('\n') {
                if !piece.is_empty() && piece.len() + line.len() + 1 > max {
                    chunks.push(std::mem::take(&mut piece));
                }
                if !piece.is_empty() {
                    piece.push('\n');
                }
                piece.push_str(line);
                // 单行仍超长: 按字符硬切
                while piece.len() > max {
                    let cut = char_cut(&piece, max);
                    let rest = piece[cut..].to_string();
                    piece.truncate(cut);
                    chunks.push(std::mem::take(&mut piece));
                    piece = rest;
                }
            }
            if !piece.is_empty() {
                chunks.push(piece);
            }
            continue;
        }
        if !current.is_empty() && current.len() + block.len() + 2 > max {
            push_current(&mut chunks, &mut current);
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(block);
    }
    push_current(&mut chunks, &mut current);
    if chunks.is_empty() {
        chunks.push(markdown.to_string());
    }
    chunks
}

/// 在不超过 `max` 个字符的位置断开(优先句子/逗号/空格)
fn char_cut(s: &str, max: usize) -> usize {
    let chars: Vec<(usize, char)> = s.char_indices().take(max).collect();
    let boundary = chars
        .iter()
        .rposition(|(_, c)| matches!(c, '。' | '！' | '？' | '；' | '，' | '、' | ' ' | '\n'))
        .map(|i| chars[i].0 + 1)
        .filter(|&pos| pos > 0 && pos < s.len());
    match boundary {
        Some(pos) => pos,
        None => chars.last().map(|(i, c)| i + c.len_utf8()).unwrap_or(s.len()).min(s.len()),
    }
}


/// 收集目录树中所有带 spine 位置的条目
fn collect_marks(node: &TocNode, out: &mut Vec<(usize, String)>) {
    if let Some(s) = node.spine {
        out.push((s, node.label.clone()));
    }
    for child in &node.children {
        collect_marks(child, out);
    }
}

/// 把 spine 位置映射为章节序号, 填入目录树
fn attach_chapters(node: TocNode, chapter_of_spine: &[usize]) -> EpubTocEntry {
    let chapter = node.spine.and_then(|s| chapter_of_spine.get(s).copied());
    EpubTocEntry {
        label: node.label,
        depth: node.depth,
        chapter,
        children: node
            .children
            .into_iter()
            .map(|c| attach_chapters(c, chapter_of_spine))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// 图片提取
// ---------------------------------------------------------------------------

/// 扫描 markdown 中的图片引用, 把图片资源解压到临时目录
fn extract_images(
    epub: &Epub,
    markdown: &mut String,
    temp_dir: &Path,
    images: &mut HashMap<String, PathBuf>,
) {
    let urls = scan_image_urls(markdown);
    for url in urls {
        if images.contains_key(&url) {
            continue;
        }
        if let Some(path) = extract_one_image(epub, &url, temp_dir) {
            images.insert(url, path);
        }
    }
}

fn extract_one_image(epub: &Epub, url: &str, temp_dir: &Path) -> Option<PathBuf> {
    let cleaned = url.split(['#', '?']).next().unwrap_or(url);
    let me = epub.manifest().by_href(cleaned).or_else(|| {
        // 尝试按相对引用解析(去掉 .. 前缀后)
        let normalized = normalize_href(cleaned);
        if normalized != cleaned {
            epub.manifest().by_href(&normalized)
        } else {
            None
        }
    })?;
    let bytes = me.read_bytes().ok()?;
    let file_name = cleaned
        .rsplit('/')
        .next()
        .map(sanitize_file_name)
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "image".to_string());
    let out_path = temp_dir.join(&file_name);
    // 同名冲突时加序号
    let out_path = if out_path.exists() {
        let mut i = 1;
        loop {
            let candidate = temp_dir.join(format!("{i}_{file_name}"));
            if !candidate.exists() {
                break candidate;
            }
            i += 1;
        }
    } else {
        out_path
    };
    std::fs::write(&out_path, bytes).ok()?;
    Some(out_path)
}

/// 扫描 markdown 中的所有图片 url
fn scan_image_urls(markdown: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = markdown;
    while let Some(pos) = rest.find("![") {
        rest = &rest[pos + 2..];
        let Some(end_alt) = rest.find("](") else {
            break;
        };
        let rest_url = &rest[end_alt + 2..];
        let Some(end_url) = rest_url.find(')') else {
            break;
        };
        let url = rest_url[..end_url].trim().to_string();
        if !url.is_empty() && !url.starts_with("data:") {
            out.push(url);
        }
        rest = &rest_url[end_url + 1..];
    }
    out
}

/// 简单解析相对路径(去掉 ./ 与 .. 段)
fn normalize_href(href: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in href.split('/') {
        match seg {
            "." | "" => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(seg),
        }
    }
    parts.join("/")
}

fn sanitize_file_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn path_hash(path: &Path) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// XHTML → Markdown 转换
// ---------------------------------------------------------------------------

/// XHTML → Markdown 转换
pub fn xhtml_to_markdown(source: &str) -> Result<String> {
    let cleaned = sanitize_xhtml(source);
    let doc = roxmltree::Document::parse(&cleaned).context("XHTML 解析失败")?;
    let mut out = String::new();
    render_children(&doc.root_element(), &mut out);
    Ok(out)
}

/// 渲染根元素下的所有子节点(逐节点分派)
fn render_children(node: &roxmltree::Node, out: &mut String) {
    for child in node.children() {
        render_node(&child, out);
    }
}

/// 分派单个节点: 文本 → 内联; 块级元素 → 块渲染; 其他元素 → 内联渲染
fn render_node(node: &roxmltree::Node, out: &mut String) {
    if node.is_text() {
        push_text(out, node.text().unwrap_or(""));
        return;
    }
    if !node.is_element() {
        return;
    }
    let name = node.tag_name().name();
    if is_block_element(name) {
        render_block(node, out);
    } else {
        render_inline(node, out);
    }
}

/// 渲染一个块级元素本身
fn render_block(node: &roxmltree::Node, out: &mut String) {
    let name = node.tag_name().name();
    match name {
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
            ensure_blank_line(out);
            let level = name[1..].parse::<usize>().unwrap_or(1);
            for _ in 0..level {
                out.push('#');
            }
            out.push(' ');
            render_inline(node, out);
            out.push('\n');
        }
        "p" | "div" | "section" | "article" | "figcaption" | "td" | "th" => {
            ensure_blank_line(out);
            render_children(node, out);
            out.push('\n');
        }
        "blockquote" => {
            ensure_blank_line(out);
            let mut inner = String::new();
            render_children(node, &mut inner);
            for line in inner.lines() {
                out.push_str("> ");
                out.push_str(line);
                out.push('\n');
            }
        }
        "ul" | "ol" => {
            ensure_blank_line(out);
            render_list(node, out, name == "ol");
        }
        "li" => {
            ensure_blank_line(out);
            out.push_str("- ");
            render_inline(node, out);
            out.push('\n');
        }
        "pre" => {
            ensure_blank_line(out);
            out.push_str("```\n");
            let text = node.text().unwrap_or("");
            if text.trim().is_empty() {
                // <pre><code>…</code></pre>
                render_raw_text(node, out);
            } else {
                out.push_str(text);
            }
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("```\n");
        }
        "table" => {
            ensure_blank_line(out);
            render_table(node, out);
        }
        "hr" => {
            ensure_blank_line(out);
            out.push_str("---\n");
        }
        "script" | "style" | "title" | "head" | "nav" | "link" | "meta" | "svg" | "figure" => {}
        _ => {
            ensure_blank_line(out);
            render_children(node, out);
            out.push('\n');
        }
    }
}

fn is_block_element(name: &str) -> bool {
    matches!(
        name,
        "html" | "body" | "head" | "title" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "p" | "div"
            | "section" | "article" | "blockquote" | "ul" | "ol" | "li" | "pre" | "table" | "hr"
            | "figure" | "figcaption" | "td" | "th"
    )
}

/// 渲染列表
fn render_list(node: &roxmltree::Node, out: &mut String, ordered: bool) {
    let mut i = 1;
    for child in node.children() {
        if !child.is_element() {
            continue;
        }
        if child.tag_name().name() != "li" {
            continue;
        }
        let prefix = if ordered {
            let s = format!("{i}. ");
            i += 1;
            s
        } else {
            "- ".to_string()
        };
        out.push_str(&prefix);

        // li 内可能直接是文本, 也可能是内联元素 + 嵌套列表
        for c in child.children() {
            if c.is_text() {
                push_text(out, c.text().unwrap_or(""));
            } else if c.is_element() {
                let n = c.tag_name().name();
                if n == "ul" || n == "ol" {
                    let mut nested = String::new();
                    render_list(&c, &mut nested, n == "ol");
                    for line in nested.lines() {
                        out.push_str("  ");
                        out.push_str(line);
                        out.push('\n');
                    }
                } else {
                    render_inline(&c, out);
                }
            }
        }
        out.push('\n');
    }
}

/// 渲染表格(简单格式: 每行一行, 用 | 分隔)
fn render_table(node: &roxmltree::Node, out: &mut String) {
    for tr in node.children().filter(|c| c.is_element() && c.tag_name().name() == "tr") {
        let mut cells: Vec<String> = Vec::new();
        for cell in tr.children().filter(|c| c.is_element()) {
            let name = cell.tag_name().name();
            if name == "td" || name == "th" {
                let mut s = String::new();
                render_inline(&cell, &mut s);
                cells.push(s.trim().to_string());
            }
        }
        if !cells.is_empty() {
            out.push('|');
            for cell in &cells {
                out.push(' ');
                out.push_str(cell);
                out.push_str(" |");
            }
            out.push('\n');
        }
    }
}

/// 渲染内联元素(对节点本身分派)
fn render_inline(node: &roxmltree::Node, out: &mut String) {
    if node.is_text() {
        push_text(out, node.text().unwrap_or(""));
        return;
    }
    if !node.is_element() {
        return;
    }
    match node.tag_name().name() {
        "b" | "strong" => {
            out.push_str("**");
            render_inline_children(node, out);
            out.push_str("**");
        }
        "i" | "em" => {
            out.push('*');
            render_inline_children(node, out);
            out.push('*');
        }
        "code" => {
            out.push('`');
            render_inline_children(node, out);
            out.push('`');
        }
        "a" => {
            let href = node.attribute("href").unwrap_or("");
            let mut text = String::new();
            render_inline_children(node, &mut text);
            let text = text.trim();
            if text.is_empty() {
                return;
            }
            out.push('[');
            out.push_str(text);
            out.push_str("](");
            out.push_str(href);
            out.push(')');
        }
        "img" => {
            let src = node.attribute("src").unwrap_or("");
            let alt = node.attribute("alt").unwrap_or("");
            out.push_str("![");
            out.push_str(alt);
            out.push_str("](");
            out.push_str(src);
            out.push(')');
        }
        "br" => out.push('\n'),
        _ => render_inline_children(node, out),
    }
}

/// 递归渲染所有子节点(内联)
fn render_inline_children(node: &roxmltree::Node, out: &mut String) {
    for child in node.children() {
        render_inline(&child, out);
    }
}

/// 原样输出文本(用于 pre/code, 不做空白折叠)
fn render_raw_text(node: &roxmltree::Node, out: &mut String) {
    for child in node.descendants() {
        if child.is_text() {
            out.push_str(child.text().unwrap_or(""));
        }
    }
}

/// 追加文本并把连续空白折叠为单个空格
fn push_text(out: &mut String, text: &str) {
    let mut words = text.split_whitespace();
    if let Some(first) = words.next() {
        // 紧跟行内标记符(如 **)时不额外加空格, 保证中文排版紧凑
        if !out.is_empty() && !out.ends_with(['\n', ' ', '\t', '*', '`', '_', '~']) {
            out.push(' ');
        }
        out.push_str(first);
        for word in words {
            out.push(' ');
            out.push_str(word);
        }
    }
}

fn ensure_blank_line(out: &mut String) {
    if out.is_empty() {
        return;
    }
    while out.ends_with('\n') {
        out.pop();
    }
    out.push_str("\n\n");
}

/// 从 markdown 提取第一个标题作为章节名
fn first_heading(markdown: &str) -> Option<String> {
    markdown
        .lines()
        .find_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                let title = trimmed.trim_start_matches('#').trim();
                if !title.is_empty() {
                    Some(title.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
}

/// 去掉 XML 声明与 DOCTYPE, 便于 roxmltree 解析
fn sanitize_xhtml(source: &str) -> String {
    let mut s = source.to_string();
    // <?xml …?>
    if let Some(start) = s.find("<?xml") {
        if let Some(rel) = s[start..].find("?>") {
            s.replace_range(start..start + rel + 2, "");
        }
    }
    // <!DOCTYPE …> (可能含内部子集 [ … ])
    if let Some(start) = s.to_lowercase().find("<!doctype") {
        let end = if s[start..].contains('[') {
            s[start..]
                .find("]>")
                .map(|e| start + e + 2)
        } else {
            s[start..].find('>').map(|e| start + e + 1)
        };
        if let Some(end) = end {
            s.replace_range(start..end, "");
        }
    }
    s
}

/// 兜底: 简单去掉所有标签
fn strip_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut in_tag = false;
    for c in source.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if in_tag => {}
            _ => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_basic_xhtml_to_markdown() {
        let xhtml = r#"<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<h1>第一章 开始</h1>
<p>这是<strong>加粗</strong>与<em>斜体</em>的<code>代码</code>。</p>
<p>第二段包含<a href="https://example.com">链接</a>。</p>
<ul><li>项目一</li><li>项目二</li></ul>
<blockquote><p>引用文字</p></blockquote>
<pre><code>let x = 1;</code></pre>
</body>
</html>"#;
        let md = xhtml_to_markdown(xhtml).unwrap();
        assert!(md.contains("# 第一章 开始"), "{md}");
        assert!(md.contains("这是**加粗**与*斜体*的`代码`。"), "{md}");
        assert!(md.contains("[链接](https://example.com)"), "{md}");
        assert!(md.contains("- 项目一"), "{md}");
        assert!(md.contains("> 引用文字"), "{md}");
        assert!(md.contains("```\nlet x = 1;\n```"), "{md}");
    }

    #[test]
    fn collapses_whitespace_in_paragraphs() {
        let xhtml = "<html><body><p>  第一行\n  第二行  </p></body></html>";
        let md = xhtml_to_markdown(xhtml).unwrap();
        assert_eq!(md.trim(), "第一行 第二行");
    }

    #[test]
    fn first_heading_works() {
        assert_eq!(first_heading("## 标题内容\n\n正文").as_deref(), Some("标题内容"));
        assert_eq!(first_heading("正文\n# 后面的标题").as_deref(), Some("后面的标题"));
        assert_eq!(first_heading("没有标题"), None);
    }

    #[test]
    fn normalize_href_works() {
        assert_eq!(normalize_href("../images/a.png"), "images/a.png");
        assert_eq!(normalize_href("./a.png"), "a.png");
        assert_eq!(normalize_href("x/../a.png"), "a.png");
        assert_eq!(normalize_href("OEBPS/img/a.png"), "OEBPS/img/a.png");
    }
}
