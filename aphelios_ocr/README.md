# aphelios_ocr

## Dolphin OCR Notes

### Why the new two-stage streaming path can be slower

The current streaming implementation in `src/dolphin/model.rs` was intended to overlap:

- CPU crop + resize + normalize for batch `N + 1`
- GPU inference for batch `N`

In practice, the slower behavior and the larger "trough" region in the runtime curve are
consistent with three structural issues in the current code:

1. The preprocessing worker is not truly parallel on CPU.
   The code comments describe parallel preprocessing, but the worker used a regular
   iterator instead of Rayon. That means the new path paid the cost of an extra thread
   and channel without fully using CPU parallelism.

2. The worker performed `Tensor::to_device(...)` on the background thread.
   For a Metal backend, this can compete with the main inference loop for the same GPU
   command queue or synchronization path. Instead of "CPU preparing while GPU infers",
   it can become "background thread submitting GPU work while foreground thread also
   submits GPU work", which increases stalls and widens the low-utilization region.

3. The streaming batch shape differs from the old stable path.
   The old path used a smaller, fixed inference batch and processed all crops after a
   fully prepared preprocessing phase. The new path used a larger streaming batch,
   which can increase per-step latency and make utilization swings more visible.

There is also a correctness issue: reading order was previously derived from the local
index inside each chunk, so numbering could reset per chunk. That does not directly
explain the slowdown, but it is a bug in the streaming implementation.

### What should be changed first

The first round of fixes should keep the streaming structure, but separate CPU work from
GPU work more strictly:

- perform crop + resize + normalize in the worker with Rayon
- keep the worker output on CPU memory
- move the CPU-to-GPU transfer back to the inference thread just before batching
- keep global reading order across chunks

This keeps the intended pipeline overlap while reducing the chance that the background
thread interferes with Metal inference.

### Single-GPU scheduler for PDF OCR

The current implementation now favors a single-GPU scheduler over concurrent stage-1 and
stage-2 GPU work.

The scheduling rule is:

- maintain a page queue for stage 1
- maintain a clip queue for stage 2
- if stage-2 clips are at or above the batch threshold, run stage 2 first
- otherwise run a batched stage 1 to refill the clip queue
- keep only one GPU task active at a time

This is meant to match the observed runtime profile:

- PDF render is relatively small
- stage 1 is moderate
- stage 2 dominates total time
- stage 2 already benefits from larger batches

With this design, the GPU is fed by the most valuable pending work without letting two
model stages compete for the same Metal device queue.

### Should image cropping move to Metal GPU

It is possible in principle, but it is not the best first optimization target for this
module.

Reasons:

1. The current crop operation is relatively cheap compared with model encode/decode.
   Most of the cost here is usually not rectangle slicing itself, but resize,
   normalization, tensor packing, device transfer, and autoregressive decoding.

2. GPU cropping only pays off when the image is already on GPU and stays there.
   In the current pipeline, source pages start on CPU, layout parsing also begins from a
   CPU-side image object, and later stages still need prompt assembly and batching on the
   host. If we upload to GPU only for crop and then move data around again, transfer and
   synchronization overhead can erase the gain.

3. Metal crop/resize becomes attractive only if we redesign the second stage around a
   GPU-native image pipeline.
   That would typically mean:
   - keep page pixels in `MTLTexture`
   - run crop + resize + normalization in one compute pass
   - write directly into model input buffers
   - minimize host round-trips

So the recommended order is:

1. fix the current streaming pipeline
2. measure crop/normalize/device-transfer time separately
3. only then decide whether a Metal preprocessing kernel is justified

### PDF skip caveat

The current PDF skip logic avoids rerunning OCR on pages that already have
`*_page.txt` output, but it does not skip PDF rendering itself. The page is still
rendered and decoded before the skip decision is applied. If PDF resume performance
matters, the skip check must happen before page render/decode.
