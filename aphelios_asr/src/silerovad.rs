//! VAD (Voice Activity Detection) module
//!
//! Unified Silero VAD implementation combining:
//! - Robust ONNX runtime handling (ORT cache, multi-sample-rate support)
//! - Sliding window smoothing for reduced false positives
//! - Full audio pipeline (loading + resampling)

pub mod detector;
pub mod engine;

// Core engine types (VadSegment here is sample-index based)
pub use engine::{SileroVadEngine, VadConfig};

// High-level detectors (these use base::VadSegment which is f64-based)
pub use detector::{run_vad_with_path, VadDetector, VadProcessor};
