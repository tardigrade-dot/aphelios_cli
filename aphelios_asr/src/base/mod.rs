pub mod qwen3llm;

#[derive(Debug, Clone)]
pub struct VadSegment {
    pub start: f64,
    pub end: f64,
    pub avg_prob: f32,
}

impl VadSegment {
    pub fn new(start: f64, end: f64, avg_prob: f32) -> Self {
        Self {
            start,
            end,
            avg_prob,
        }
    }
}

#[derive(Debug)]
pub struct AudioBatch {
    pub start: f64,
    pub end: f64,
    pub duration: f64,
    pub segments_count: usize,
}

#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct AsrSegment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
    pub sub_segments: Vec<SubSegment>,
}

#[derive(Debug, Clone)]
pub struct SubSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}
