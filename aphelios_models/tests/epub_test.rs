use std::fs::File;

use anyhow::{Ok, Result};
use aphelios_core::init_logging;
use boko::TextRange;
use futures_util::io::BufWriter;
use rbook::Epub;

#[test]
fn epub_test() -> Result<()> {
    init_logging();
    let epub_path =
        "/Users/larry/Downloads/自由的窄廊國家與社會如何決定自由的命運 = The Narrow Corridor States, Societies, and the Fate of Liberty (戴倫 · 艾塞默魯, Daron Acemoglu, 詹姆斯 · 羅賓森 etc.) (z-library.sk, 1lib.sk, z-lib.sk).epub";
    let book =
        Epub::open(epub_path).unwrap_or_else(|e| panic!("open epub failed: {} {}", epub_path, e));
    for data_result in book.reader() {
        let data = data_result.unwrap();
        let kind = data.manifest_entry().kind();
        assert_eq!("application/xhtml+xml", kind.as_str());
        assert_eq!("xhtml", kind.subtype());

        // Print the readable content
        println!("{}", data.content());
    }
    Ok(())
}

#[test]
fn epub_test2() -> Result<()> {
    use boko::Book;
    init_logging();

    let epub_path =
        "/Users/larry/Downloads/自由的窄廊國家與社會如何決定自由的命運 = The Narrow Corridor States, Societies, and the Fate of Liberty (戴倫 · 艾塞默魯, Daron Acemoglu, 詹姆斯 · 羅賓森 etc.) (z-library.sk, 1lib.sk, z-lib.sk).epub";

    // Open a book (format auto-detected from extension)
    let mut book = Book::open(epub_path)?;

    // Access metadata
    println!("Title: {}", book.metadata().title);
    println!("Authors: {:?}", book.metadata().authors);

    // Iterate chapters
    let spine: Vec<_> = book.spine().to_vec();
    for entry in spine {
        let chapter = book.load_chapter(entry.id)?;
        let node_count = chapter.node_count();
        println!("node_count: {}", node_count);
    }

    let file = File::create("output.md")?;

    let _ = book.export(boko::Format::Markdown, &mut &file);
    Ok(())
}
