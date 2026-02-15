//! Constants for Demucs model

pub struct Constants;

impl Constants {
    pub const SAMPLE_RATE: u32 = 44100;
    pub const FFT_SIZE: usize = 4096;
    pub const HOP_SIZE: usize = 1024;
    pub const TRAINING_SAMPLES: usize = 343980;
    pub const MODEL_SPEC_BINS: usize = 2048;
    pub const MODEL_SPEC_FRAMES: usize = 336;
    pub const SEGMENT_OVERLAP: f32 = 0.25;
    pub const TRACKS: [&'static str; 4] = ["drums", "bass", "other", "vocals"];

    // Default model URL (Hugging Face Hub)
    pub const DEFAULT_MODEL_URL: &'static str = 
        "https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx";
}