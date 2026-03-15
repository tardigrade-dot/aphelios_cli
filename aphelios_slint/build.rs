fn main() {
    // 使用 SLINT_OUTPUT_GENERATED 环境变量来指定输出文件
    // 所有 UI 文件将合并到一个文件中

    let config = slint_build::CompilerConfiguration::new()
        .embed_resources(slint_build::EmbedResourcesKind::EmbedFiles);

    // 编译所有 UI 文件
    slint_build::compile_with_config("assets/main_menu.slint", config.clone()).unwrap();
    slint_build::compile_with_config("assets/ocr_ui.slint", config.clone()).unwrap();
    slint_build::compile_with_config("assets/asr_ui.slint", config.clone()).unwrap();
    slint_build::compile_with_config("assets/tts_ui.slint", config.clone()).unwrap();
    slint_build::compile_with_config("assets/settings_ui.slint", config).unwrap();
}
