use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tokio::sync::broadcast;
use tracing::info;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Get the log directory path
/// For release builds: uses the application's data directory
/// For local/dev builds: uses current directory's logs folder
fn get_log_dir() -> PathBuf {
    // Try to get the executable's directory for release builds
    if let Some(exe_path) = std::env::current_exe().ok() {
        if let Some(exe_dir) = exe_path.parent() {
            // Check if we're running from an .app bundle (macOS)
            #[cfg(target_os = "macos")]
            {
                if exe_path.to_string_lossy().contains(".app/Contents/MacOS") {
                    // Use ~/Library/Logs for macOS apps
                    if let Some(home) = dirs::home_dir() {
                        let log_dir = home.join("Library").join("Logs").join("aphelios");
                        let _ = fs::create_dir_all(&log_dir);
                        return log_dir;
                    }
                }
            }

            // For other release builds, use executable's directory
            let log_dir = exe_dir.join("logs");
            let _ = fs::create_dir_all(&log_dir);
            return log_dir;
        }
    }

    // Fallback: use current directory's logs folder (for local/dev)
    let log_dir = PathBuf::from("logs");
    let _ = fs::create_dir_all(&log_dir);
    log_dir
}

/// Initialize logging with both file and console output
/// Automatically detects build type and configures appropriate output
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

        // Create file appender for release builds
        let log_dir = get_log_dir();
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let log_file = log_dir.join(format!("aphelios_{}.log", timestamp));

        // Try to create file appender, but don't fail if it can't
        let file_appender =
            tracing_appender::rolling::never(&log_dir, format!("aphelios_{}.log", timestamp));
        let file_layer = fmt::layer()
            .with_writer(file_appender)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .compact();

        // Use both layers: console (fmt_layer) and file (file_layer)
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(file_layer)
            .try_init();

        // Log initialization
        tracing::info!("Logging initialized. Log file: {:?}", log_file);
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

#[cfg(feature = "profiling")]
pub fn init_chrome_logging() -> Option<tracing_chrome::FlushGuard> {
    use chrono::Local;
    use tracing_chrome::ChromeLayerBuilder;

    let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let log_dir = get_log_dir();
    let trace_file = log_dir.join(format!("trace_profile_{}.json", ts));
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off"));

    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .include_args(true)
        .file(trace_file)
        .build();

    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .pretty();

    // 使用 try_init 避免多次初始化报错
    let is_init = tracing_subscriber::registry()
        .with(filter)
        .with(chrome_layer)
        .with(fmt_layer)
        .try_init()
        .is_ok();

    if is_init {
        Some(guard)
    } else {
        None
    }
}

/// 全局广播通道，用于分发日志消息
static LOG_TX: OnceLock<broadcast::Sender<String>> = OnceLock::new();

#[cfg(feature = "profiling")]
static CHROME_GUARD: OnceLock<Mutex<tracing_chrome::FlushGuard>> = OnceLock::new();

/// 获取广播通道的发送端
fn get_log_tx() -> &'static broadcast::Sender<String> {
    LOG_TX.get_or_init(|| {
        let (tx, _) = broadcast::channel(1000);
        tx
    })
}

/// 订阅日志消息
pub fn subscribe_logs() -> broadcast::Receiver<String> {
    get_log_tx().subscribe()
}

/// 一个简单的 Writer，将写入的内容发送到广播通道
struct BroadcastWriter;

impl std::io::Write for BroadcastWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(s) = std::str::from_utf8(buf) {
            // 发送日志消息（去掉末尾换行符，因为 Slint 的 Text 会自动处理）
            let msg = s.trim_end().to_string();
            if !msg.is_empty() {
                let _ = get_log_tx().send(msg);
            }
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> fmt::MakeWriter<'a> for BroadcastWriter {
    type Writer = BroadcastWriter;
    fn make_writer(&self) -> Self::Writer {
        BroadcastWriter
    }
}

/// 初始化带广播支持的日志系统
pub fn init_slint_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,ort=off,h2=off,hyper=off,candle_core=off"));

        // 1. 标准输出层（带颜色，输出到终端）
        let stdout_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true)
            .pretty();

        // 2. Slint 广播层（不带颜色，发送到广播通道）
        let slint_layer = fmt::layer()
            .with_target(false)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false)
            .with_ansi(false) // 禁用颜色码，防止 UI 显示乱码
            .with_writer(BroadcastWriter);

        // 3. 文件输出层（带颜色，输出到文件）
        let log_dir = get_log_dir();
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let log_file = log_dir.join(format!("aphelios_{}.log", timestamp));

        // Try to create file appender, but don't fail if it can't
        let file_appender =
            tracing_appender::rolling::never(&log_dir, format!("aphelios_{}.log", timestamp));
        let file_layer = fmt::layer()
            .with_writer(file_appender)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .compact();

        // 4. Chrome 性能跟踪层（仅在 profiling 特性启用时）
        #[cfg(feature = "profiling")]
        let chrome_layer = {
            use tracing_chrome::ChromeLayerBuilder;
            let trace_file = log_dir.join(format!("trace_profile_{}.json", timestamp));
            let (chrome_layer, guard) = ChromeLayerBuilder::new()
                .include_args(true)
                .file(trace_file)
                .build();
            // 存储 guard 到全局变量，确保它在程序生命周期内存活
            let _ = CHROME_GUARD.set(Mutex::new(guard));
            chrome_layer
        };

        // 构建注册表，条件添加 Chrome 层
        let registry = tracing_subscriber::registry()
            .with(filter)
            .with(stdout_layer)
            .with(slint_layer)
            .with(file_layer);

        #[cfg(feature = "profiling")]
        let registry = registry.with(chrome_layer);

        let _ = registry.try_init();

        info!("Logging initialized. Log file: {:?}", log_file);
        #[cfg(feature = "profiling")]
        info!(
            "Chrome tracing enabled. Trace file: {:?}",
            log_dir.join(format!("trace_profile_{}.json", timestamp))
        );
    });
}
