slint::include_modules!();

use anyhow::Result;
use aphelios_cli::{cli::Cli, run};
use clap::Parser;
use std::{thread, time::Duration};

#[tokio::main]
async fn main() -> Result<(), slint::PlatformError> {
    println!("Hello, world!");

    // run_cli().await
    run_app()
}

async fn run_cli() {
    let cli = Cli::parse();

    if let Err(err) = run(cli).await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run_app() -> Result<(), slint::PlatformError> {
    // slint::register_font_from_path("assets/pingfang-sc-regular.ttf")?;
    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    // 注册 UI 回调
    ui.on_start_search(move |query| {
        let ui_handle = ui_weak.unwrap();

        // 1. 设置 UI 为加载状态
        ui_handle.set_is_loading(true);
        ui_handle.set_search_results(slint::ModelRc::from(vec![].as_slice()));

        let ui_weak_for_thread = ui_weak.clone();
        let query_str = query.to_string();

        // 2. 启动模拟后端的线程
        thread::spawn(move || {
            println!("开始检索: {}", query_str);

            // 模拟 Embedding + USearch 的耗时操作
            thread::sleep(Duration::from_secs(2));

            // 模拟搜索结果
            let mock_results = if query_str.contains("苏联") {
                vec![
                    "《二手时间》.epub".into(),
                    "《切尔诺贝利的祭祷》.pdf".into(),
                ]
            } else {
                vec!["《深入理解 Rust》.pdf".into()]
            };

            // 3. 将结果回传给 UI 线程
            // Slint 的 invoke_from_event_loop 是跨线程更新 UI 的安全方式
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(ui) = ui_weak_for_thread.upgrade() {
                    ui.set_is_loading(false);
                    let results_model: Vec<slint::SharedString> =
                        mock_results.into_iter().map(|s: String| s.into()).collect();
                    ui.set_search_results(slint::ModelRc::from(results_model.as_slice()));
                }
            });
        });
    });

    ui.run()
}
