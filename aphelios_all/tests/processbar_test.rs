use indicatif::{ProgressBar, ProgressStyle};
use tracing::{debug, info};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// 假设你使用了 anyhow 或类似的错误处理库
#[test]
fn process_bar_test() -> anyhow::Result<()> {
    // 1. 创建 IndicatifLayer
    let indicatif_layer = IndicatifLayer::new();

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .init();

    let pb = ProgressBar::new(100);

    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")?
            .progress_chars("##-"),
    );

    for i in 0..100 {
        info!("processing item {}", i);
        std::thread::sleep(std::time::Duration::from_millis(50));
        pb.inc(1);
    }

    pb.finish_with_message("done");

    Ok(())
}
