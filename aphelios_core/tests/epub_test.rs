use anyhow::Result;
use aphelios_core::init_logging;
use rbook::Epub;

#[test]
fn epub_test() -> Result<()> {
    init_logging();
    let epub_path =
        "/Users/larry/Downloads/谈谈方法 (笛卡尔) (z-library.sk, 1lib.sk, z-lib.sk).epub";
    let book = Epub::open(epub_path)?;
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
