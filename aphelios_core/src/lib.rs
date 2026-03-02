//! Aphelios Core - 音频处理核心库
//!
//! 提供语音识别 (ASR)、语音活动检测 (VAD)、音频分离 (Demucs)、
//! 说话人分离 (DIA) 等音频处理功能。
//!
//! # 模块
//!
//! - [`audio`] - 音频加载、重采样、保存
//! - [`vad`] - 语音活动检测
//! - [`asr`] - 语音识别
//! - [`demucs`] - 音频分离
//! - [`dia`] - 说话人分离
//! - [`utils`] - 通用工具（日志、计时器）

pub mod asr;
pub mod audio;
pub mod demucs;
pub mod dia;
pub mod utils;
pub mod vad;

// 重新导出常用类型
pub use audio::{AudioLoader, AudioSaver, Resampler};
pub use utils::{ScopedTimer, Timer, init_logging};
