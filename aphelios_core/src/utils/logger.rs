//! 日志初始化工具
//!
//! 提供统一的日志配置和初始化功能

use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

/// 初始化日志系统
///
/// 支持通过 RUST_LOG 环境变量控制日志级别
/// 默认级别为 info
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

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
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
