//! VAD (Voice Activity Detection) 模块
//! 
//! 使用 Silero VAD 模型进行语音活动检测

mod detector;
mod types;

pub use detector::{VadDetector, VadConfig};
pub use types::{VadSegment, VadResult};
