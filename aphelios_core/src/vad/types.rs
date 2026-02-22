//! VAD 数据类型

/// VAD 检测到的语音片段
#[derive(Debug, Clone)]
pub struct VadSegment {
    /// 开始时间（秒）
    pub start: f64,
    /// 结束时间（秒）
    pub end: f64,
    /// 持续时间（秒）
    pub duration: f64,
    /// 平均语音概率
    pub avg_probability: f32,
}

impl VadSegment {
    pub fn new(start: f64, end: f64, avg_probability: f32) -> Self {
        Self {
            start,
            end,
            duration: end - start,
            avg_probability,
        }
    }
}

/// VAD 检测结果
#[derive(Debug, Clone)]
pub struct VadResult {
    /// 输入音频时长（秒）
    pub audio_duration: f64,
    /// 检测到的语音片段
    pub segments: Vec<VadSegment>,
    /// 总语音时长（秒）
    pub total_speech_duration: f64,
    /// 语音占比
    pub speech_ratio: f32,
}

impl VadResult {
    pub fn new(audio_duration: f64, segments: Vec<VadSegment>) -> Self {
        let total_speech_duration: f64 = segments.iter().map(|s| s.duration).sum();
        let speech_ratio = if audio_duration > 0.0 {
            (total_speech_duration / audio_duration) as f32
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

    /// 合并相邻的片段（小于指定间隔的合并为一个）
    pub fn merge_adjacent(&mut self, max_gap: f64) {
        if self.segments.len() <= 1 {
            return;
        }

        let mut merged = Vec::new();
        let mut current = self.segments[0].clone();

        for segment in self.segments.iter().skip(1) {
            let gap = segment.start - current.end;
            if gap <= max_gap {
                // 合并
                current.end = segment.end;
                current.duration = current.end - current.start;
                current.avg_probability = ((current.avg_probability as f64 * current.duration
                    + segment.avg_probability as f64 * segment.duration)
                    / (current.duration + segment.duration)) as f32;
            } else {
                merged.push(current);
                current = segment.clone();
            }
        }

        merged.push(current);
        self.segments = merged;
        
        // 重新计算统计
        self.total_speech_duration = self.segments.iter().map(|s| s.duration).sum();
        self.speech_ratio = if self.audio_duration > 0.0 {
            (self.total_speech_duration / self.audio_duration) as f32
        } else {
            0.0
        };
    }
}
