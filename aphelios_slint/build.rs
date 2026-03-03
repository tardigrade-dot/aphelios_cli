// fn main() {
//     slint_build::compile_with_config(
//         "assets/ocr_ui.slint",
//         slint_build::CompilerConfiguration::new()
//             .embed_resources(slint_build::EmbedResourcesKind::EmbedForSoftwareRenderer),
//     )
//     .unwrap();
// }

// build.rs
fn main() {
    slint_build::compile_with_config(
        "assets/ocr_ui.slint",
        slint_build::CompilerConfiguration::new()
            // 改为 Embed，这样硬件和软件渲染器都能用
            .embed_resources(slint_build::EmbedResourcesKind::EmbedFiles),
    )
    .unwrap();

    // 确保字体文件变化时触发重新编译
    // println!("cargo:rerun-if-changed=assets/pingfang-sc-regular.ttf");
}
