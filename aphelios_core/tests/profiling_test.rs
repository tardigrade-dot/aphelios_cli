use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;
use tracing_tree::HierarchicalLayer;

/// 生成 Chrome 性能追踪文件的测试
///
/// 运行方式:
///   RUST_LOG=info cargo test --features profiling test_chrome_profiling -- --nocapture
///
/// 生成的 trace 文件位于: ./trace_profile.json
/// 在 Chrome 中打开: chrome://tracing -> Load -> 选择 trace_profile.json
#[test]
fn test_chrome_profiling() -> Result<()> {
    // 输出文件路径
    #[cfg(feature = "profiling")]
    let trace_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("trace_profile.json");

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off"));

    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .pretty();

    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .include_args(true)
        .file(trace_file.to_str().unwrap())
        .build();

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(chrome_layer)
        .init();

    // 示例：追踪多个嵌套的 span
    {
        let _span = tracing::info_span!("outer_operation", name = "外层操作").entered();
        info!("外层操作开始");

        // 模拟一些工作
        std::thread::sleep(Duration::from_millis(100));

        {
            let _span = tracing::info_span!("inner_operation_1", name = "内层操作1").entered();
            info!("内层操作1开始");
            std::thread::sleep(Duration::from_millis(50));
        } // inner_operation_1 结束

        {
            let _span = tracing::info_span!("inner_operation_2", name = "内层操作2").entered();
            info!("内层操作2开始");
            std::thread::sleep(Duration::from_millis(80));
        } // inner_operation_2 结束

        // 循环操作示例
        for i in 0..3 {
            let _span = tracing::info_span!("loop_iteration", iteration = i).entered();
            info!("循环迭代 {}", i);
            std::thread::sleep(Duration::from_millis(30));
        }
        {
            fake_method("hello")
        }

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        info!("input: {}", input);
        info!("外层操作完成");
    } // outer_operation 结束

    // 保持 guard 存活直到测试结束，确保数据写入文件
    drop(guard);

    println!("\n\nChrome trace 文件已生成: {}", trace_file.display());
    println!("在 Chrome 中打开: chrome://tracing -> Load -> 选择 trace_profile.json\n");

    Ok(())
}

#[test]
fn test_chrome_profiling2() -> Result<()> {
    // 输出文件路径
    #[cfg(feature = "profiling")]
    let subscriber = Registry::default().with(HierarchicalLayer::new(2));
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off"));
    // tracing::subscriber::set_global_default(subscriber).unwrap();

    // let fmt_layer = fmt::layer()
    //     .with_target(true)
    //     .with_thread_ids(true)
    //     .with_file(true)
    //     .with_line_number(true)
    //     .pretty();m

    tracing_subscriber::registry()
        .with(filter)
        // .with(fmt_layer)
        // .with(tree_layer)
        .with(HierarchicalLayer::new(2))
        .init();

    // 示例：追踪多个嵌套的 span
    {
        let _span = tracing::info_span!("outer_operation", name = "外层操作").entered();
        info!("外层操作开始");

        // 模拟一些工作
        std::thread::sleep(Duration::from_millis(100));

        {
            let _span = tracing::info_span!("inner_operation_1", name = "内层操作1").entered();
            info!("内层操作1开始");
            std::thread::sleep(Duration::from_millis(50));
        } // inner_operation_1 结束

        {
            let _span = tracing::info_span!("inner_operation_2", name = "内层操作2").entered();
            info!("内层操作2开始");
            std::thread::sleep(Duration::from_millis(80));
        } // inner_operation_2 结束

        // 循环操作示例
        for i in 0..3 {
            let _span = tracing::info_span!("loop_iteration", iteration = i).entered();
            info!("循环迭代 {}", i);
            std::thread::sleep(Duration::from_millis(30));
        }
        {
            fake_method("hello")
        }

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        info!("input: {}", input);
        info!("外层操作完成");
    } // outer_operation 结束
    Ok(())
}

#[cfg(feature = "profiling")]
#[tracing::instrument(name = "fake_method")]
fn fake_method(para: &str) {
    std::thread::sleep(Duration::from_millis(800));
}
