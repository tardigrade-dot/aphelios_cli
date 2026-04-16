use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use zip::ZipArchive;

#[derive(Debug, Clone)]
pub struct Chapter {
    pub title: String,
    pub href: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct EpubBook {
    pub title: String,
    pub chapters: Vec<Chapter>,
}

impl EpubBook {
    /// 加载并解析 EPUB 文件
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("无法打开 EPUB 文件: {}", path.as_ref().display()))?;
        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)
            .with_context(|| "无法解析 EPUB ZIP 格式")?;

        // 读取容器文件获取根目录
        let rootfile = Self::find_rootfile(&mut archive)?;

        // 读取 OPF 文件获取元数据和内容
        let (title, manifest, spine) = Self::parse_opf(&mut archive, &rootfile)?;

        // 获取所有章节内容
        let chapters = Self::extract_chapters(&mut archive, &manifest, &spine)?;

        Ok(Self { title, chapters })
    }

    fn find_rootfile(archive: &mut ZipArchive<BufReader<File>>) -> Result<String> {
        // 首先尝试标准的 container.xml 位置
        let container_xml = archive.by_name("META-INF/container.xml")?;
        let container_content = Self::read_to_string(container_xml)?;

        // 解析 container.xml 找到根文件路径
        // 简化解析：查找 rootfile 的 full-path 属性
        for line in container_content.lines() {
            if line.contains("rootfile") {
                if let Some(start) = line.find("full-path=\"") {
                    let start = start + 12;
                    if let Some(end) = line[start..].find('"') {
                        let mut path = line[start..start + end].to_string();
                        // URL 解码
                        path = Self::url_decode(&path);
                        return Ok(path);
                    }
                }
            }
        }

        anyhow::bail!("无法在 container.xml 中找到根文件路径")
    }

    fn parse_opf(
        archive: &mut ZipArchive<BufReader<File>>,
        rootfile: &str,
    ) -> Result<(String, Vec<(String, String)>, Vec<String>)> {
        // URL 解码路径
        let rootfile_decoded = Self::url_decode(rootfile);

        // 尝试多种可能的路径格式
        let try_paths = Self::get_opf_try_paths(&rootfile_decoded);

        let mut opf_content_opt = None;
        for try_path in &try_paths {
            if let Ok(opf_file) = archive.by_name(try_path) {
                opf_content_opt = Some(Self::read_to_string(opf_file)?);
                break;
            }
        }

        let opf_content = opf_content_opt
            .with_context(|| format!("无法在归档中找到 OPF 文件: {}", rootfile_decoded))?;

        let opf_dir = Path::new(&rootfile_decoded)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut title = String::from("未知标题");
        let mut manifest: Vec<(String, String)> = Vec::new();
        let mut spine_ids: Vec<String> = Vec::new();

        let mut in_metadata = false;

        for line in opf_content.lines() {
            let line = line.trim();

            // 检测 metadata 区段开始
            if line.contains("<metadata") || line.contains("<metadata ") {
                in_metadata = true;
            } else if line.contains("</metadata>") {
                in_metadata = false;
            }

            // 提取标题（从 dc:title 或 metadata 中的 Title）
            if in_metadata {
                if line.contains("<dc:title") || line.contains("<title") {
                    if let Some(content) = Self::extract_tag_content(line, "title") {
                        title = content;
                    } else if let Some(content) = Self::extract_tag_content(line, "dc:title") {
                        title = content;
                    }
                }
            }

            // 解析 manifest 项
            if line.contains("<item") && line.contains("id=\"") && line.contains("href=\"") {
                let id = Self::extract_attribute(line, "id");
                let href = Self::extract_attribute(line, "href");
                if let (Some(id), Some(href)) = (id, href) {
                    // URL 解码 href
                    let href_decoded = Self::url_decode(&href);
                    // 构建完整路径
                    let full_href = if opf_dir.is_empty() {
                        href_decoded
                    } else {
                        format!("{}/{}", opf_dir, href_decoded)
                    };
                    manifest.push((id, full_href));
                }
            }

            // 解析 spine 项
            if line.contains("<itemref") && line.contains("idref=\"") {
                if let Some(idref) = Self::extract_attribute(line, "idref") {
                    spine_ids.push(idref);
                }
            }
        }

        Ok((title, manifest, spine_ids))
    }

    fn get_opf_try_paths(opf_path: &str) -> Vec<String> {
        let mut paths = vec![opf_path.to_string()];

        // 尝试不同的前缀
        let prefixes = ["OEBPS/", "Text/", "Content/", "OPS/", ""];
        let filename = Path::new(opf_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| opf_path.to_string());

        for prefix in prefixes {
            if !opf_path.starts_with(prefix) && !prefix.is_empty() {
                paths.push(format!("{}{}", prefix, opf_path));
                paths.push(format!("{}{}", prefix, filename));
            }
        }
        paths.push(filename);

        // 去重
        paths.sort();
        paths.dedup();
        paths
    }

    fn extract_chapters(
        archive: &mut ZipArchive<BufReader<File>>,
        manifest: &[(String, String)],
        spine_ids: &[String],
    ) -> Result<Vec<Chapter>> {
        let mut chapters = Vec::new();
        let manifest_map: std::collections::HashMap<&str, &str> =
            manifest.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

        for idref in spine_ids {
            if let Some(href) = manifest_map.get(idref.as_str()) {
                // 尝试读取文件
                let try_paths = Self::get_content_try_paths(href);

                let mut content = String::new();
                let mut found = false;

                for try_path in &try_paths {
                    // URL 解码路径
                    let decoded_path = Self::url_decode(try_path);
                    if let Ok(file) = archive.by_name(&decoded_path) {
                        content = Self::read_to_string(file)?;
                        found = true;
                        break;
                    }
                    // 尝试原始路径
                    if !decoded_path.eq(try_path) {
                        if let Ok(file) = archive.by_name(try_path) {
                            content = Self::read_to_string(file)?;
                            found = true;
                            break;
                        }
                    }
                }

                if found {
                    // 提取标题（从第一个 h1 或 title 标签）
                    let chapter_title = Self::extract_title_from_html(&content)
                        .unwrap_or_else(|| format!("第 {} 章", chapters.len() + 1));

                    // 清理 HTML 标签，保留纯文本
                    let clean_content = Self::strip_html_tags(&content);

                    chapters.push(Chapter {
                        title: chapter_title,
                        href: href.to_string(),
                        content: clean_content,
                    });
                }
            }
        }

        Ok(chapters)
    }

    fn get_content_try_paths(href: &str) -> Vec<String> {
        let mut paths = vec![href.to_string()];

        // 尝试不同的目录前缀
        let prefixes = ["OEBPS/", "Text/", "Content/", "OPS/"];
        let filename = Path::new(href)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| href.to_string());

        for prefix in prefixes {
            if !href.starts_with(prefix) {
                paths.push(format!("{}{}", prefix, href));
                paths.push(format!("{}{}", prefix, filename));
            }
        }
        paths.push(filename);

        // 去重
        paths.sort();
        paths.dedup();
        paths
    }

    fn url_decode(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '%' {
                let mut hex = String::new();
                for _ in 0..2 {
                    if let Some(ch) = chars.next() {
                        hex.push(ch);
                    }
                }
                if hex.len() == 2 {
                    if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                        result.push(byte as char);
                    } else {
                        result.push_str("%");
                        result.push_str(&hex);
                    }
                } else {
                    result.push('%');
                    result.push_str(&hex);
                }
            } else if c == '+' {
                result.push(' ');
            } else {
                result.push(c);
            }
        }

        result
    }

    fn read_to_string<R: Read>(reader: R) -> Result<String> {
        let mut s = String::new();
        let mut reader = reader;
        reader.read_to_string(&mut s)?;
        Ok(s)
    }

    fn extract_tag_content(line: &str, tag: &str) -> Option<String> {
        // 处理自闭合标签如 <dc:title>内容</dc:title>
        let start_tag = format!("<{}", tag);

        if let Some(start) = line.find(&start_tag) {
            let after_tag = &line[start..];
            if let Some(content_start) = after_tag.find('>') {
                let content = &after_tag[content_start + 1..];
                if let Some(end_pos) = content.find('<') {
                    return Some(content[..end_pos].trim().to_string());
                }
            }
        }
        None
    }

    fn extract_attribute(line: &str, attr: &str) -> Option<String> {
        let marker = format!("{}=\"", attr);
        if let Some(start) = line.find(&marker) {
            let after_attr = &line[start + marker.len()..];
            if let Some(end) = after_attr.find('"') {
                return Some(after_attr[..end].to_string());
            }
        }
        None
    }

    fn extract_title_from_html(html: &str) -> Option<String> {
        // 尝试提取 <title> 标签
        for tag in &["title", "h1", "h2", "h3"] {
            let start_tag = format!("<{}", tag);
            if let Some(start) = html.find(&start_tag) {
                let after = &html[start..];
                if let Some(content_start) = after.find('>') {
                    let content = &after[content_start + 1..];
                    if let Some(end) = content.find('<') {
                        let title = content[..end].trim().to_string();
                        if !title.is_empty() {
                            return Some(title);
                        }
                    }
                }
            }
        }

        None
    }

    fn strip_html_tags(html: &str) -> String {
        let mut result = String::with_capacity(html.len());
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;
        let mut last_was_space = false;

        let chars: Vec<char> = html.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            // 检查是否进入 script/style 标签
            if i + 7 < chars.len() {
                let tag_start: String = chars[i..i + 7].iter().collect();
                if tag_start.to_lowercase() == "<script" {
                    in_script = true;
                } else if tag_start.to_lowercase() == "</scrip" {
                    in_script = false;
                    i += 8;
                    continue;
                }
            }

            if i + 6 < chars.len() {
                let tag_start: String = chars[i..i + 6].iter().collect();
                if tag_start.to_lowercase() == "<style" {
                    in_style = true;
                } else if tag_start.to_lowercase() == "</styl" {
                    in_style = false;
                    i += 7;
                    continue;
                }
            }

            if in_script || in_style {
                i += 1;
                continue;
            }

            if c == '<' {
                in_tag = true;
                i += 1;
                continue;
            } else if c == '>' && in_tag {
                in_tag = false;
                i += 1;
                continue;
            }

            if in_tag {
                i += 1;
                continue;
            }

            // 处理普通字符
            if c.is_whitespace() {
                if !last_was_space && !result.is_empty() {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(c);
                last_was_space = false;
            }

            i += 1;
        }

        // 清理多余空格
        let mut cleaned = String::new();
        let mut last_was_space = false;
        for c in result.chars() {
            if c == ' ' {
                if !last_was_space {
                    cleaned.push(c);
                    last_was_space = true;
                }
            } else {
                cleaned.push(c);
                last_was_space = false;
            }
        }

        cleaned.trim().to_string()
    }

    /// 获取总页数（每章为一页）
    pub fn total_chapters(&self) -> usize {
        self.chapters.len()
    }

    /// 获取指定章节内容
    pub fn get_chapter(&self, index: usize) -> Option<&Chapter> {
        self.chapters.get(index)
    }
}

pub struct EpubLogic {
    book: Mutex<Option<EpubBook>>,
}

impl EpubLogic {
    pub fn new() -> Self {
        Self {
            book: Mutex::new(None),
        }
    }

    pub fn load_book(&self, path: &str) -> Result<EpubBook> {
        let book = EpubBook::from_path(path)?;
        let mut lock = self.book.lock().unwrap();
        *lock = Some(book.clone());
        Ok(book)
    }

    pub fn get_chapter_content(&self, index: usize) -> Option<String> {
        let lock = self.book.lock().unwrap();
        lock.as_ref()?.get_chapter(index).map(|c| c.content.clone())
    }

    pub fn get_book_title(&self) -> Option<String> {
        let lock = self.book.lock().unwrap();
        lock.as_ref().map(|b| b.title.clone())
    }
}

use std::sync::Mutex;
