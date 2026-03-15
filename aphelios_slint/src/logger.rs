use std::sync::OnceLock;
use tokio::sync::broadcast;
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// 全局广播通道，用于分发日志消息
static LOG_TX: OnceLock<broadcast::Sender<String>> = OnceLock::new();

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

        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(stdout_layer)
            .with(slint_layer)
            .try_init();
    });
}
