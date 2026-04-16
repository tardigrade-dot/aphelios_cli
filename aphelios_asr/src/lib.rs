pub mod granite;
pub mod qwen3llm;
pub mod qwenasr;
pub mod sensevoice;
pub mod silerovad;
pub mod text_match;
pub mod whisper;

/// Voice activity detection segment (time in milliseconds)
#[derive(Debug, Clone)]
pub struct VadSegment {
    pub start: i64,
    pub end: i64,
    pub avg_prob: f64,
}

impl VadSegment {
    pub fn new(start: i64, end: i64, avg_prob: f64) -> Self {
        Self {
            start,
            end,
            avg_prob,
        }
    }
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

/// Batch of aggregated segments for downstream processing.
#[derive(Debug, Clone)]
pub struct AudioBatch {
    pub start: f64,
    pub end: f64,
    pub duration: f64,
    pub segments_count: usize,
}

/// VAD detection result for a full audio file.
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Input audio duration in seconds.
    pub audio_duration: f64,
    /// Detected speech segments.
    pub segments: Vec<VadSegment>,
    /// Total speech duration in seconds.
    pub total_speech_duration: f64,
    /// Speech ratio (0.0 - 1.0).
    pub speech_ratio: f32,
}

impl VadResult {
    pub fn new(audio_duration: f64, segments: Vec<VadSegment>) -> Self {
        let total_speech_duration: f64 = segments.iter().map(|s| (s.end - s.start) as f64).sum();
        let speech_ratio = if audio_duration > 0.0 {
            total_speech_duration as f32 / audio_duration as f32
        } else {
            0.0
        };

        Self {
            audio_duration,
            segments,
            total_speech_duration,
            speech_ratio,
        }
    }

    /// Merge adjacent segments with gap smaller than `max_gap` (in milliseconds).
    pub fn merge_adjacent(&mut self, max_gap: i64) {
        if self.segments.len() <= 1 {
            return;
        }

        let mut merged = Vec::new();
        let mut current = self.segments[0].clone();

        for seg in self.segments.iter().skip(1) {
            let gap = seg.start.saturating_sub(current.end);
            if gap <= max_gap {
                if seg.end > current.end {
                    current.end = seg.end;
                }
            } else {
                merged.push(current);
                current = seg.clone();
            }
        }
        merged.push(current);
        self.segments = merged;

        self.total_speech_duration = self.segments.iter().map(|s| (s.end - s.start) as f64).sum();
        self.speech_ratio = if self.audio_duration > 0.0 {
            self.total_speech_duration as f32 / self.audio_duration as f32
        } else {
            0.0
        };
    }
}
