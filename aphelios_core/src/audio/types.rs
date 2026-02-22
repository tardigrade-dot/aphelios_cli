//! 音频数据类型定义

/// 单声道音频缓冲区
#[derive(Debug, Clone)]
pub struct MonoBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl MonoBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self { samples, sample_rate }
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// 立体声音频缓冲区
#[derive(Debug, Clone)]
pub struct StereoBuffer {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
}

impl StereoBuffer {
    pub fn new(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        Self { left, right, sample_rate }
    }

    /// 从单声道创建伪立体声
    pub fn from_mono(mono: &MonoBuffer) -> Self {
        Self {
            left: mono.samples.clone(),
            right: mono.samples.clone(),
            sample_rate: mono.sample_rate,
        }
    }

    /// 转换为单声道（取左右声道平均值）
    pub fn to_mono(&self) -> MonoBuffer {
        let samples: Vec<f32> = self.left
            .iter()
            .zip(self.right.iter())
            .map(|(&l, &r)| (l + r) / 2.0)
            .collect();
        MonoBuffer::new(samples, self.sample_rate)
    }

    pub fn duration_secs(&self) -> f64 {
        self.left.len() as f64 / self.sample_rate as f64
    }

    pub fn len(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    /// 确保是立体声（如果只有一个声道则复制）
    pub fn ensure_stereo(&mut self) {
        if self.right.is_empty() {
            self.right = self.left.clone();
        }
    }
}

/// 通用音频缓冲区（支持单声道和立体声）
#[derive(Debug, Clone)]
pub enum AudioBuffer {
    Mono(MonoBuffer),
    Stereo(StereoBuffer),
}

impl AudioBuffer {
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Mono(m) => m.sample_rate,
            Self::Stereo(s) => s.sample_rate,
        }
    }

    pub fn to_stereo(self) -> StereoBuffer {
        match self {
            Self::Mono(m) => StereoBuffer::from_mono(&m),
            Self::Stereo(s) => s,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        match self {
            Self::Mono(m) => m.duration_secs(),
            Self::Stereo(s) => s.duration_secs(),
        }
    }
}
