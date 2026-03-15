use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_chrome_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off"));

        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true)
            .pretty();

        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
            .include_args(true)
            // .file("trace.json")
            .build();

        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(chrome_layer)
            .try_init();
    })
}

pub fn init_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off"));

        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true)
            .pretty();

        let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(chrome_layer)
            .try_init();
    });
}

/// 初始化日志系统（简化版，用于测试）
pub fn init_test_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        let filter = EnvFilter::new("info,ort=off,h2=off,hyper=off,candle=off");

        let fmt_layer = fmt::layer()
            .with_target(false)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false)
            .compact();

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
    });
}
