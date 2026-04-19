use ndarray::{Array1, ArrayD, Axis};
use rand::Rng;
use crate::moss_tts_nano::config::Manifest;

pub fn argmax(values: &Array1<f32>) -> usize {
    values
        .iter()
        .enumerate()
        .fold(0, |max_idx, (idx, &val)| {
            if val > values[max_idx] {
                idx
            } else {
                max_idx
            }
        })
}

pub fn softmax(values: &mut Array1<f32>) {
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for x in values.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    for x in values.iter_mut() {
        *x /= sum;
    }
}

pub fn apply_repetition_penalty(
    values: &mut Array1<f32>,
    previous_token_ids: &[i32],
    repetition_penalty: f32,
) {
    if repetition_penalty == 1.0 || previous_token_ids.is_empty() {
        return;
    }

    let mut seen = std::collections::HashSet::new();
    for &id in previous_token_ids {
        seen.insert(id as usize);
    }

    for &id in &seen {
        if id < values.len() {
            let val = values[id];
            if val < 0.0 {
                values[id] = val * repetition_penalty;
            } else {
                values[id] = val / repetition_penalty;
            }
        }
    }
}

pub fn sample_from_scores<R: Rng>(
    values: &Array1<f32>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    rng: &mut R,
) -> usize {
    let mut scores = values.to_owned();

    if temperature > 0.0 {
        scores /= temperature;
    }

    if top_k > 0 && (top_k as usize) < scores.len() {
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let threshold = indexed_scores[top_k as usize - 1].1;
        for x in scores.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }

    if top_p > 0.0 && top_p < 1.0 {
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut probs = Array1::from_iter(indexed_scores.iter().map(|&(_, s)| s));
        softmax(&mut probs);

        let mut cumulative_prob = 0.0;
        let mut cut_off = indexed_scores.len();
        for (i, &p) in probs.iter().enumerate() {
            cumulative_prob += p;
            if cumulative_prob > top_p {
                cut_off = i + 1;
                break;
            }
        }

        for i in cut_off..indexed_scores.len() {
            scores[indexed_scores[i].0] = f32::NEG_INFINITY;
        }
    }

    softmax(&mut scores);

    let mut r: f32 = rng.gen();
    for (i, &p) in scores.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }

    argmax(&scores)
}

pub fn sample_assistant_text_token<R: Rng>(
    text_logits: &Array1<f32>,
    manifest: &Manifest,
    do_sample: bool,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    rng: &mut R,
) -> i32 {
    let candidate_ids = vec![
        manifest.tts_config.audio_assistant_slot_token_id,
        manifest.tts_config.audio_end_token_id,
    ];

    let mut candidate_scores = Array1::zeros(candidate_ids.len());
    for (i, &id) in candidate_ids.iter().enumerate() {
        candidate_scores[i] = text_logits[id as usize];
    }

    if !do_sample {
        return candidate_ids[argmax(&candidate_scores)];
    }

    let sampled_idx = sample_from_scores(&candidate_scores, temperature, top_k, top_p, rng);
    candidate_ids[sampled_idx]
}

pub fn sample_audio_token<R: Rng>(
    audio_logits: &Array1<f32>,
    previous_token_ids: &[i32],
    do_sample: bool,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    rng: &mut R,
) -> i32 {
    let mut scores = audio_logits.to_owned();
    apply_repetition_penalty(&mut scores, previous_token_ids, repetition_penalty);

    if !do_sample {
        return argmax(&scores) as i32;
    }

    sample_from_scores(&scores, temperature, top_k, top_p, rng) as i32
}
