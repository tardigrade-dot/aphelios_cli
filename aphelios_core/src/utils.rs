//! 通用工具模块
//!
//! 提供日志、计时器、错误处理等通用功能

pub mod common;
pub mod logger;
pub mod progress;
pub mod timer;
pub mod token_output_stream;

pub use logger::init_logging;
pub use progress::AppProgressBar;

pub fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
