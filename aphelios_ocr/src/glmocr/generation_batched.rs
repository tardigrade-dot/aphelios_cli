//! Batched inference for GLM-OCR: process multiple region images in a single model call.
//!
//! The text decoder (attention, rotary embedding, KV-cache) already supports batch
//! dimensions throughout. This module batches N region crops through a single prefill
//! + autoregressive decode, giving near-linear speedup for multi-region documents.

use anyhow::Result;
use candle_core::IndexOp;
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;

use crate::glmocr::image_processor::preprocess_image;
use crate::glmocr::model::GlmOcrModel;
use crate::glmocr::tokenizer::GlmOcrTokenizer;
use crate::glmocr::MAX_MERGED_PATCHES;

/// Generate text from multiple region images in a single batched model call.
///
/// Each image + prompt pair is processed independently but batched through the
/// text decoder for efficient GPU/accelerator utilization.
pub fn generate_batched(
    model: &GlmOcrModel,
    tokenizer: &GlmOcrTokenizer,
    images: &[DynamicImage],
    prompts: &[&str],
    max_tokens: usize,
) -> Result<Vec<String>> {
    let batch = images.len();
    if batch == 0 {
        return Ok(Vec::new());
    }
    if batch != prompts.len() {
        anyhow::bail!(
            "generate_batched: {} images but {} prompts",
            batch,
            prompts.len()
        );
    }

    let device = &model.device;
    let pad_token_id = tokenizer.pad_token_id();

    // =========================================================================
    // Phase 1: Preprocess each region independently
    // =========================================================================
    tracing::info!(
        "generate_batched: batch={}, max_tokens={}",
        batch,
        max_tokens
    );
    let mut per_region = Vec::with_capacity(batch);
    let mut max_seq_len = 0usize;

    for (i, (img, prompt)) in images.iter().zip(prompts.iter()).enumerate() {
        // Warn if over patch budget (scale_to_patch_budget rounds to unit=28,
        // so a small overshoot is expected). Proceed regardless since the vision
        // encoder output determines the actual token count.
        let unit = 28u32;
        let w_merged = ((img.width() + unit / 2) / unit).max(1);
        let h_merged = ((img.height() + unit / 2) / unit).max(1);
        if w_merged * h_merged > MAX_MERGED_PATCHES {
            tracing::warn!(
                "Region {i} has {} merged patches (max {MAX_MERGED_PATCHES}) — proceeding",
                w_merged * h_merged
            );
        }

        // 1a. Preprocess image
        let (pixel_values, grid_thw) = preprocess_image(img, &model.config.vision_config, device)?;
        let pixel_values = pixel_values.to_dtype(model.dtype)?;

        // 1b. Vision encoder → image embeddings
        let image_embeds = model.vision_encoder.forward(&pixel_values, grid_thw)?;
        let num_image_tokens = image_embeds.dim(0)?;

        // 1c. Build input token sequence
        let input_ids = tokenizer.build_input_ids(prompt, num_image_tokens)?;
        let seq_len = input_ids.len();
        max_seq_len = max_seq_len.max(seq_len);

        // 1d. Embed + merge vision and text
        let inputs_embeds = model.embed_and_merge(&input_ids, &image_embeds)?;
        // 1e. Compute 3D position IDs
        let (position_ids, next_pos) = model.compute_3d_positions(&input_ids, grid_thw)?;

        per_region.push(RegionData {
            inputs_embeds, // [1, seq_len, hidden]
            position_ids,  // [3, 1, seq_len]
            next_pos,
            effective_len: seq_len,
            grid_thw,
        });
    }

    // =========================================================================
    // Phase 2: Pad + stack into batched tensors
    // =========================================================================
    tracing::info!(
        "Phase 2: padding {} regions to max_seq_len={}",
        batch,
        max_seq_len
    );
    let hidden_size = per_region[0].inputs_embeds.dim(2)?;

    let mut batched_embeds_data = vec![0.0f32; batch * max_seq_len * hidden_size];
    let mut batched_pos_data = vec![0i64; 3 * batch * max_seq_len];

    for (i, r) in per_region.iter().enumerate() {
        let sl = r.effective_len;

        // Copy inputs_embeds [1, sl, hidden] → batched position
        let emb = r.inputs_embeds.to_vec3::<f32>()?;
        for j in 0..sl {
            for k in 0..hidden_size {
                let dst = i * max_seq_len * hidden_size + j * hidden_size + k;
                batched_embeds_data[dst] = emb[0][j][k];
            }
        }

        // Copy position_ids [3, 1, sl] → batched position
        let pos = r.position_ids.to_vec3::<i64>()?;
        for d in 0..3 {
            for j in 0..sl {
                let dst = d * batch * max_seq_len + i * max_seq_len + j;
                batched_pos_data[dst] = pos[d][0][j];
            }
        }
    }

    let batched_inputs_embeds = Tensor::from_vec(
        batched_embeds_data,
        (batch, max_seq_len, hidden_size),
        device,
    )?
    .to_dtype(model.dtype)?;

    let batched_position_ids = Tensor::from_vec(batched_pos_data, (3, batch, max_seq_len), device)?;

    // =========================================================================
    // Phase 3: Batched attention mask
    // =========================================================================
    tracing::info!(
        "Phase 3: building batched causal mask [{batch}, 1, {max_seq_len}, {max_seq_len}]"
    );
    let attention_mask = build_batched_causal_mask(&per_region, max_seq_len, device)?;

    // =========================================================================
    // Phase 4: Batched prefill
    // =========================================================================
    tracing::info!("Phase 4: batched prefill");
    let kv_caches = Vec::new();
    let (hidden_states, mut kv_caches) = model.text_decoder.forward_to_hidden(
        &batched_inputs_embeds,
        &batched_position_ids,
        Some(&attention_mask),
        kv_caches,
    )?;

    // Gather logits from the last REAL token per batch element (not padding)
    let gather_positions: Vec<usize> = per_region
        .iter()
        .map(|r| r.effective_len.saturating_sub(1))
        .collect();

    let mut hidden_at_pos = Vec::with_capacity(batch);
    for i in 0..batch {
        let h = hidden_states.i(i)?.narrow(0, gather_positions[i], 1)?; // [1, hidden]
        hidden_at_pos.push(h);
    }
    let last_hidden = Tensor::stack(&hidden_at_pos, 0)?; // [batch, 1, hidden]
    let logits = model.text_decoder.lm_head_forward(&last_hidden)?; // [batch, 1, vocab_size]

    // Get first token for each batch element (greedy)
    let mut next_tokens = batch_argmax(&logits)?; // [batch]
    let mut output_tokens: Vec<Vec<u32>> = (0..batch).map(|_| Vec::new()).collect();
    let mut finished = vec![false; batch];
    let mut current_positions: Vec<i64> = per_region.iter().map(|r| r.next_pos).collect();

    tracing::info!(
        "Prefill done. Starting autoregressive decode (max {} steps)",
        max_tokens
    );

    // =========================================================================
    // Phase 5: Batched autoregressive decode
    // =========================================================================
    for step in 0..max_tokens {
        if step % 100 == 0 {
            tracing::info!(
                "Decode step {}/{} (finished: {:?})",
                step,
                max_tokens,
                finished
            );
        }
        // Check each element for EOS
        for i in 0..batch {
            if finished[i] {
                continue;
            }
            if tokenizer.is_eos(next_tokens[i]) {
                finished[i] = true;
                continue;
            }
            output_tokens[i].push(next_tokens[i]);
        }

        // All finished?
        if finished.iter().all(|&f| f) {
            break;
        }

        // Embed next token for each batch element
        let token_ids_i64: Vec<i64> = (0..batch)
            .map(|i| {
                if finished[i] {
                    pad_token_id as i64
                } else {
                    next_tokens[i] as i64
                }
            })
            .collect();

        let token_tensor = Tensor::from_vec(token_ids_i64, (batch, 1), device)?;
        let token_embeds = model.text_decoder.embed(&token_tensor)?; // [batch, 1, hidden]

        // Position IDs: all 3 dims get the same position value per element
        let mut pos_vals = vec![0i64; 3 * batch];
        for i in 0..batch {
            let p = current_positions[i];
            pos_vals[i] = p; // dim 0, element i
            pos_vals[batch + i] = p; // dim 1, element i
            pos_vals[2 * batch + i] = p; // dim 2, element i
        }
        let pos = Tensor::from_vec(pos_vals, (3, batch, 1), device)?;

        // Single-token decode (no attention mask needed)
        let (hidden, new_caches) =
            model
                .text_decoder
                .forward_to_hidden(&token_embeds, &pos, None, kv_caches)?;
        kv_caches = new_caches;

        // Apply lm_head to get logits [batch, 1, vocab_size]
        let logits = model.text_decoder.lm_head_forward(&hidden)?;

        // Get next token for each unfinished element
        // For finished elements, use pad_token (won't be used)
        let logits_vec = logits.squeeze(1)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for i in 0..batch {
            if !finished[i] {
                let (best_idx, _) = logits_vec[i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap();
                next_tokens[i] = best_idx as u32;
                current_positions[i] += 1;
            }
        }
    }

    tracing::info!(
        "Decode done in {} steps. Decoding {} sequences to text.",
        max_tokens,
        batch
    );

    // =========================================================================
    // Phase 6: Decode per-sequence tokens
    // =========================================================================
    let mut results = Vec::with_capacity(batch);
    for tokens in output_tokens {
        let text = tokenizer.decode(&tokens, true)?;
        results.push(text);
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Data accumulated per region during preprocessing.
struct RegionData {
    inputs_embeds: Tensor, // [1, seq_len, hidden_size]
    position_ids: Tensor,  // [3, 1, seq_len]
    next_pos: i64,
    effective_len: usize,
    #[allow(dead_code)]
    grid_thw: [u32; 3],
}

/// Build a batched causal attention mask with padding.
///
/// For each batch element i with `effective_len[i]`:
///   - positions [0..eff, 0..eff] have a standard causal mask
///   - positions >= eff are masked to -inf (both query and key sides)
///
/// Returns: [batch, 1, max_seq_len, max_seq_len]
fn build_batched_causal_mask(
    regions: &[RegionData],
    max_seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let batch = regions.len();
    let mut mask_data = vec![f32::NEG_INFINITY; batch * max_seq_len * max_seq_len];

    for i in 0..batch {
        let eff = regions[i].effective_len;
        let offset = i * max_seq_len * max_seq_len;
        for j in 0..eff {
            for k in 0..eff {
                if k <= j {
                    // Causal: query j can attend to key k ≤ j
                    mask_data[offset + j * max_seq_len + k] = 0.0;
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        mask_data,
        (batch, 1, max_seq_len, max_seq_len),
        device,
    )?)
}

/// Get argmax for each batch element from a [batch, 1, vocab_size] logits tensor.
fn batch_argmax(logits: &Tensor) -> Result<Vec<u32>> {
    let logits = logits.squeeze(1)?; // [batch, vocab_size]
    let indices = logits.argmax(1)?; // [batch]
    let vals = indices.to_vec1::<u32>()?;
    Ok(vals)
}
