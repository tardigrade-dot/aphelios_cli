use indicatif::{ProgressBar, ProgressStyle};
use tracing::{debug, info};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::util::SubscriberInitExt;

// 假设你使用了 anyhow 或类似的错误处理库
#[test]
fn process_bar_test() -> anyhow::Result<()> {
    // 1. 创建 IndicatifLayer
    let indicatif_layer = IndicatifLayer::new();

    // 2. 注册 Subscriber
    // 注意：.init() 如果被多次调用会 panic，实际项目中建议在 main 统一初始化
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer())) // 关键：让日志输出不干扰进度条
        .with(indicatif_layer)
        .init();

    // 3. 重点：如何获取 ProgressBar？
    // indicatif_layer 并没有 ppro 字段，你需要手动创建一个 ProgressBar
    // 并通过关联将其交给 tracing 管理
    let pb = indicatif::ProgressBar::new(100);

    // 4. 设置样式
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")?
            .progress_chars("##-"),
    );

    // 5. 将进度条关联到当前的 tracing span（可选，但推荐）
    // 或者直接使用 pb，tracing-indicatif 会自动处理渲染冲突

    for i in 0..100 {
        // 使用 info! 会通过 tracing-indicatif 打印，不会搞乱进度条
        info!("processing item {}", i);

        std::thread::sleep(std::time::Duration::from_millis(50));
        pb.inc(1);
    }

    pb.finish_with_message("done");

    Ok(())
}
