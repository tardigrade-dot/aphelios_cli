fn main() {
    let config = slint_build::CompilerConfiguration::new()
        .embed_resources(slint_build::EmbedResourcesKind::EmbedFiles);

    // 编译主窗口（自动导入所有页面组件）
    slint_build::compile_with_config("assets/app_window.slint", config).unwrap();
}
