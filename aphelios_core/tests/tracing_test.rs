use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

#[test]
fn test_test() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off"));

    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .pretty();

    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .include_args(true)
        // .file("trace.json")
        .build();

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(chrome_layer)
        .init();

    info!("test tracing chrome");

    drop(guard);
    Ok(())
}

#[test]
fn tes2_test() -> anyhow::Result<()> {
    let (chrome_layer, _) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
    info!("test tracing chrome");
    // static INIT: std::sync::Once = std::sync::Once::new();
    // let guard = INIT.call_once(|| {
    //     let filter = EnvFilter::new("info");

    //     let fmt_layer = fmt::layer();

    //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();

    //     tracing_subscriber::registry()
    //         .with(filter)
    //         .with(fmt_layer)
    //         .with(chrome_layer)
    //         .init();
    //     guard
    // });
    // info!("test tracing chrome");
    // drop(guard); // 强制 flush
    Ok(())
}
