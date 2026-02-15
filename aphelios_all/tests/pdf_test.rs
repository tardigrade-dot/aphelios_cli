use std::sync::Arc;

use anyhow::{Error as e, Result};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{render, RenderSettings};
use image::DynamicImage;

#[test]
fn pdf_hayro_test() -> anyhow::Result<()> {
    let pdf_path = "/Users/larry/github.com/colab-script2/data_src/extracted_pages.pdf";
    let file = std::fs::read(pdf_path)?;
    let pdf = Pdf::new(Arc::new(file)).unwrap();

    let mut images: Vec<DynamicImage> = Vec::new();

    let interpreter_settings = InterpreterSettings::default();
    let render_settings = RenderSettings {
        bg_color: WHITE, // 建议显式设置背景色，否则透明 PDF 可能会变黑
        ..Default::default()
    };

    let mut i = 0;
    for page in pdf.pages().iter() {
        let pixmap = render(page, &interpreter_settings, &render_settings);

        // --- 核心修改点 ---
        // 1. 将 Pixmap 转换为 PNG 编码的内存数据 (Vec<u8>)
        let png_bytes = pixmap
            .into_png()
            .map_err(|e| anyhow::anyhow!("PNG encoding failed: {:?}", e))?;

        // 2. 利用 image 库直接从内存字节加载为 DynamicImage
        let img = image::load_from_memory(&png_bytes)?;

        i += 1;
        let _ = img.save(format!(
            "/Users/larry/coderesp/aphelios_cli/output/{}_page.png",
            i
        ));
        images.push(img);
    }

    Ok(())
}
