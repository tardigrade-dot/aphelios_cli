//! VAD postprocessor implementing the state machine logic from fireredvad.
//!
//! This module handles smoothing, state machine decisions, and segment extraction
//! from raw speech probabilities.

/// VAD Postprocessor implementing the FireRedVAD decision logic.
#[derive(Debug, Clone)]
pub struct VadPostprocessor {
    pub smooth_window_size: usize,
    pub prob_threshold: f32,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
}

impl VadPostprocessor {
    /// Create a new VAD postprocessor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        smooth_window_size: usize,
        prob_threshold: f32,
        min_speech_frame: usize,
        max_speech_frame: usize,
        min_silence_frame: usize,
        merge_silence_frame: usize,
        extend_speech_frame: usize,
    ) -> Self {
        Self {
            smooth_window_size: smooth_window_size.max(1),
            prob_threshold,
            min_speech_frame,
            max_speech_frame,
            min_silence_frame,
            merge_silence_frame,
            extend_speech_frame,
        }
    }

    /// Process raw probabilities into VAD decisions.
    ///
    /// # Arguments
    /// * `probs` - Raw per-frame speech probabilities
    ///
    /// # Returns
    /// Binary decisions (0=silence, 1=speech) for each frame
    pub fn process(&self, probs: &[f32]) -> Vec<i32> {
        if probs.is_empty() {
            return vec![];
        }

        // Smooth probabilities
        let smoothed = self.smooth(probs);

        // Convert to binary decisions
        let binary: Vec<i32> = smoothed
            .iter()
            .map(|&p| if p >= self.prob_threshold { 1 } else { 0 })
            .collect();

        // Apply state machine
        let mut decisions = self.state_machine(&binary);

        // Fix start transitions
        decisions = self.fix_start(&decisions);

        // Merge short silence gaps
        decisions = self.merge_silence(&decisions);

        // Extend speech boundaries
        decisions = self.extend_speech(&decisions);

        // Split overly long segments
        decisions = self.split_long(&decisions, probs);

        decisions
    }

    /// Convert decisions to time segments.
    ///
    /// # Arguments
    /// * `decisions` - Binary decisions per frame
    /// * `wav_dur` - Optional total waveform duration for clamping the last segment
    ///
    /// # Returns
    /// Vector of (start, end) tuples in seconds
    pub fn decisions_to_segments(
        &self,
        decisions: &[i32],
        wav_dur: Option<f32>,
    ) -> Vec<(f32, f32)> {
        if decisions.is_empty() {
            return vec![];
        }

        let mut segments = Vec::new();
        let mut speech_start: Option<usize> = None;

        for (t, &d) in decisions.iter().enumerate() {
            if d == 1 && speech_start.is_none() {
                speech_start = Some(t);
            } else if d == 0 && speech_start.is_some() {
                let start = speech_start.unwrap();
                segments.push((start as f32 * FRAME_SHIFT_S, t as f32 * FRAME_SHIFT_S));
                speech_start = None;
            }
        }

        // Handle trailing speech
        if let Some(start) = speech_start {
            let mut end = decisions.len() as f32 * FRAME_SHIFT_S + FRAME_LENGTH_S;
            if let Some(dur) = wav_dur {
                end = end.min(dur);
            }
            segments.push((start as f32 * FRAME_SHIFT_S, end));
        }

        // Round to 3 decimal places
        segments
            .into_iter()
            .map(|(s, e)| ((s * 1000.0).round() / 1000.0, (e * 1000.0).round() / 1000.0))
            .collect()
    }

    /// Smooth probabilities with a moving average filter.
    fn smooth(&self, probs: &[f32]) -> Vec<f32> {
        if self.smooth_window_size <= 1 {
            return probs.to_vec();
        }

        let window = self.smooth_window_size;
        let mut smoothed = vec![0.0f32; probs.len()];

        // Full convolution, then trim
        for i in 0..probs.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(probs.len());
            let sum: f32 = probs[start..end].iter().sum();
            smoothed[i] = sum / (end - start) as f32;
        }

        // Handle edge cases more carefully (match Python behavior)
        for i in 0..(window - 1).min(probs.len()) {
            let sum: f32 = probs[..=i].iter().sum();
            smoothed[i] = sum / (i + 1) as f32;
        }

        smoothed
    }

    /// State machine for VAD decisions.
    ///
    /// States:
    /// - SILENCE (0): Currently in silence
    /// - POSSIBLE_SPEECH (1): Detected potential speech start
    /// - SPEECH (2): Confirmed speech
    /// - POSSIBLE_SILENCE (3): Detected potential silence end
    fn state_machine(&self, binary: &[i32]) -> Vec<i32> {
        const SILENCE: i32 = 0;
        const POSSIBLE_SPEECH: i32 = 1;
        const SPEECH: i32 = 2;
        const POSSIBLE_SILENCE: i32 = 3;

        let mut decisions = vec![0i32; binary.len()];
        let mut state = SILENCE;
        let mut speech_start: i32 = -1;
        let mut silence_start: i32 = -1;

        for (t, &is_speech) in binary.iter().enumerate() {
            match state {
                SILENCE => {
                    if is_speech == 1 {
                        state = POSSIBLE_SPEECH;
                        speech_start = t as i32;
                    }
                }
                POSSIBLE_SPEECH => {
                    if is_speech == 1 {
                        if (t as i32 - speech_start) >= self.min_speech_frame as i32 {
                            state = SPEECH;
                            for j in (speech_start as usize)..t {
                                decisions[j] = 1;
                            }
                        }
                    } else {
                        state = SILENCE;
                        speech_start = -1;
                    }
                }
                SPEECH => {
                    if is_speech == 0 {
                        state = POSSIBLE_SILENCE;
                        silence_start = t as i32;
                    }
                }
                POSSIBLE_SILENCE => {
                    if is_speech == 0 {
                        if (t as i32 - silence_start) >= self.min_silence_frame as i32 {
                            state = SILENCE;
                            speech_start = -1;
                        }
                    } else {
                        state = SPEECH;
                        silence_start = -1;
                    }
                }
                _ => unreachable!("Invalid state: {}", state),
            }

            decisions[t] = if state == SPEECH || state == POSSIBLE_SILENCE {
                1
            } else {
                0
            };
        }

        decisions
    }

    /// Fix start transitions by extending speech backwards.
    fn fix_start(&self, decisions: &[i32]) -> Vec<i32> {
        let mut new = decisions.to_vec();

        for (t, &d) in decisions.iter().enumerate() {
            if t > 0 && decisions[t - 1] == 0 && d == 1 {
                let start = t.saturating_sub(self.smooth_window_size);
                for j in start..t {
                    new[j] = 1;
                }
            }
        }

        new
    }

    /// Merge short silence gaps between speech segments.
    fn merge_silence(&self, decisions: &[i32]) -> Vec<i32> {
        if self.merge_silence_frame == 0 {
            return decisions.to_vec();
        }

        let mut new = decisions.to_vec();
        let mut silence_start: Option<usize> = None;

        for (t, &d) in decisions.iter().enumerate() {
            if t > 0 && decisions[t - 1] == 1 && d == 0 && silence_start.is_none() {
                silence_start = Some(t);
            } else if t > 0 && decisions[t - 1] == 0 && d == 1 && silence_start.is_some() {
                let start = silence_start.unwrap();
                if t - start < self.merge_silence_frame {
                    for j in start..t {
                        new[j] = 1;
                    }
                }
                silence_start = None;
            }
        }

        new
    }

    /// Extend speech boundaries by a fixed number of frames.
    fn extend_speech(&self, decisions: &[i32]) -> Vec<i32> {
        if self.extend_speech_frame == 0 {
            return decisions.to_vec();
        }

        // Simple dilation using a sliding window
        let k = self.extend_speech_frame;
        let mut result = vec![0i32; decisions.len()];

        for (i, &d) in decisions.iter().enumerate() {
            if d == 1 {
                // Extend in both directions
                let start = i.saturating_sub(k);
                let end = (i + k + 1).min(decisions.len());
                for j in start..end {
                    result[j] = 1;
                }
            }
        }

        result
    }

    /// Split overly long speech segments at low-probability points.
    fn split_long(&self, decisions: &[i32], probs: &[f32]) -> Vec<i32> {
        let mut new = decisions.to_vec();

        // Get current segments
        let segments = self.decisions_to_segments(decisions, None);

        for (s_s, e_s) in segments {
            let sf = (s_s / FRAME_SHIFT_S) as usize;
            let ef = ((e_s / FRAME_SHIFT_S) as usize).min(probs.len());

            if ef > sf && ef - sf > self.max_speech_frame {
                let seg_probs: Vec<f32> = probs[sf..ef].to_vec();
                let splits = self.find_splits(&seg_probs);

                for split in splits {
                    if sf + split < new.len() {
                        new[sf + split] = 0;
                    }
                }
            }
        }

        new
    }

    /// Find split points for long segments.
    fn find_splits(&self, probs: &[f32]) -> Vec<usize> {
        let mut splits = Vec::new();
        let mut start = 0;
        let l = probs.len();

        while start < l {
            if l - start <= self.max_speech_frame {
                break;
            }

            let ws = start + self.max_speech_frame / 2;
            let we = (start + self.max_speech_frame).min(l);

            if ws < we && ws < probs.len() {
                let search_range = &probs[ws..we];
                let min_idx = search_range
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                splits.push(ws + min_idx);
                start = splits.last().unwrap() + 1;
            } else {
                break;
            }
        }

        splits
    }
}

// Constants for segment conversion
const FRAME_LENGTH_S: f32 = 0.025;
const FRAME_SHIFT_S: f32 = 0.010;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth() {
        let pp = VadPostprocessor::new(5, 0.4, 20, 2000, 20, 0, 0);
        let probs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let smoothed = pp.smooth(&probs);
        assert_eq!(smoothed.len(), probs.len());
        // First value should be average of [0.1] = 0.1
        assert!((smoothed[0] - 0.1).abs() < 1e-5);
        // Middle values should be smoothed
        assert!(smoothed[4] > 0.3 && smoothed[4] < 0.7);
    }

    #[test]
    fn test_decisions_to_segments() {
        let pp = VadPostprocessor::new(5, 0.4, 20, 2000, 20, 0, 0);
        // Create decisions: silence (0-9), speech (10-19), silence (20-29)
        let mut decisions = vec![0i32; 30];
        for i in 10..20 {
            decisions[i] = 1;
        }

        let segments = pp.decisions_to_segments(&decisions, None);
        assert_eq!(segments.len(), 1);
        // Speech starts at frame 10 = 0.1s, ends at frame 20 = 0.2s + 0.025s
        assert!((segments[0].0 - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_empty_input() {
        let pp = VadPostprocessor::new(5, 0.4, 20, 2000, 20, 0, 0);
        let result = pp.process(&[]);
        assert!(result.is_empty());
    }
}
