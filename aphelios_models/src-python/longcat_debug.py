import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


def batch_tokenize(tokenizer_path: str, texts: list[str], pad_id: int = 0):
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
    encodings = tokenizer.encode_batch(texts)
    max_len = max(len(enc.ids) for enc in encodings)
    input_ids = []
    attention_mask = []
    for enc in encodings:
        ids = enc.ids
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LongCat-AudioDiT debug/dump")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--prompt_audio", type=str, default=None)
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--repo_dir", type=str, default="/Users/larry/coderesp/LongCat-AudioDiT")
    parser.add_argument("--tokenizer_dir", type=str, default="/Volumes/sw/pretrained_models/umt5-base")
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--guidance_strength", type=float, default=4.0)
    parser.add_argument("--guidance_method", type=str, default="cfg", choices=["cfg", "apg"])
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"LongCat repo not found: {repo_dir}")
    sys.path.insert(0, str(repo_dir))

    from audiodit import AudioDiTModel
    from utils import approx_duration_from_text, load_audio, normalize_text
    from longcat_rng import LongCatRng

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # torch.manual_seed(args.seed) # We'll use LongCatRng for y0 instead
    if torch.mps.is_available():
        torch.mps.manual_seed(args.seed)

    root_rng = LongCatRng(args.seed)

    model = AudioDiTModel.from_pretrained(args.model_dir).to(device)
    model.vae.to_half()
    model.eval()

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    text = normalize_text(args.text)
    no_prompt = args.prompt_audio is None
    if not no_prompt:
        prompt_text = normalize_text(args.prompt_text)
        full_text = f"{prompt_text} {text}"
    else:
        prompt_text = None
        full_text = text

    input_ids, attention_mask = batch_tokenize(args.tokenizer_dir, [full_text], pad_id=0)

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    tensors["input_ids"] = input_ids.numpy()
    tensors["attention_mask"] = attention_mask.numpy()

    # 1. Text Encoder output
    with torch.no_grad():
        # Hook into the first block output
        first_block_out = None
        def hook_fn(module, input, output):
            nonlocal first_block_out
            first_block_out = output[0] if isinstance(output, tuple) else output
        
        handle = model.text_encoder.encoder.block[0].register_forward_hook(hook_fn)
        
        output = model.text_encoder(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
        )
        handle.remove()
        
    tensors["py_emb_out"] = output.hidden_states[0].detach().cpu().numpy()
    tensors["py_first_block_out"] = first_block_out.detach().cpu().numpy()
    tensors["py_last_hidden"] = output.last_hidden_state.detach().cpu().numpy()

    text_condition = model.encode_text(input_ids.to(device), attention_mask.to(device))
    tensors["text_condition"] = text_condition.detach().cpu().numpy()

    # 2. VAE Encoding (if prompt)
    if not no_prompt:
        pw = load_audio(args.prompt_audio, sr)
        off = 3
        if pw.shape[-1] % full_hop != 0:
            pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        pw = F.pad(pw, (0, full_hop * off))
        with torch.no_grad():
            # In LongCat, vae.encode returns a LatentOutput with .latent_dist
            vae_out = model.vae.encode(pw.unsqueeze(0).to(device))
            prompt_latent_vae = vae_out.latent_dist.mode()
        
        # Scale and permute to match model.forward
        latent_cond = prompt_latent_vae / model.config.scale
        if off:
            latent_cond = latent_cond[..., :-off]
        latent_cond = latent_cond.permute(0, 2, 1)
        
        prompt_dur = latent_cond.shape[1]
        prompt_latent = latent_cond.clone()
        tensors["prompt_latent"] = prompt_latent.detach().cpu().numpy()
        tensors["latent_cond"] = latent_cond.detach().cpu().numpy()
    else:
        prompt_latent = None
        prompt_dur = 0

    # Duration calculation
    prompt_time = prompt_dur * full_hop / sr
    dur_sec = approx_duration_from_text(text, max_duration=max_duration - prompt_time)
    if prompt_text is not None:
        approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
        ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
        dur_sec = dur_sec * ratio
    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))
    tensors["duration"] = np.array([duration])

    # 3. Initial Noise y0
    y0_rng = root_rng.fork(0x5930_5F4E_4F49_5345)
    y0 = y0_rng.standard_normal_tensor((1, duration, model.config.latent_dim), device=device)
    tensors["y0"] = y0.detach().cpu().numpy()

    # Hook for modulation parameters
    modulation_params = {}
    def hook(m, i, o):
        # adaln_out
        adaln_out = i[0]
        gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(adaln_out, 6, dim=-1)
        modulation_params["scale_sa"] = scale_sa.detach().cpu().numpy()
        modulation_params["shift_sa"] = shift_sa.detach().cpu().numpy()

    handle = model.transformer.blocks[0].register_forward_hook(hook)
# 4. Transformer Velocity at t=0
t_zero = torch.tensor(0.0, device=device)
total_duration = duration

if not no_prompt:
    gen_len = total_duration - latent_len
    latent_cond = F.pad(prompt_latent, (0, 0, 0, gen_len))
    empty_latent_cond = torch.zeros_like(latent_cond)
    prompt_noise = y0[:, :latent_len].clone()
else:
    latent_cond = torch.zeros(1, total_duration, model.config.latent_dim, device=device)
    empty_latent_cond = latent_cond
    prompt_noise = None

    
    tensors["latent_cond"] = latent_cond.detach().cpu().numpy()
    tensors["empty_latent_cond"] = empty_latent_cond.detach().cpu().numpy()

    # Capture intermediate transformer states
    with torch.no_grad():
        x_in = y0.clone()
        if not no_prompt:
            x_in[:, :latent_len] = prompt_noise * (1-t_zero) + latent_cond[:, :latent_len] * t_zero
        
        # We want to see the first block output inside the transformer
        block0_out = None
        def block_hook(m, i, o):
            nonlocal block0_out
            block0_out = o
        
        handle = model.transformer.blocks[0].register_forward_hook(block_hook)
        
        output = model.transformer(
            x=x_in, text=text_condition, text_len=attention_mask.sum(dim=1).to(device), time=t_zero,
            mask=mask, cond_mask=text_mask,
            return_ith_layer=model.config.repa_dit_layer, latent_cond=latent_cond,
        )
        handle.remove()
        
    tensors["py_block0_out"] = block0_out.detach().cpu().numpy()
    tensors["py_transformer_out"] = output["last_hidden_state"].detach().cpu().numpy()

    # We need to replicate the fn logic inside model.forward
    def evaluate_velocity(t, x):
        if not no_prompt:
            x[:, :latent_len] = prompt_noise * (1-t) + latent_cond[:, :latent_len] * t
        
        output = model.transformer(
            x=x, text=text_condition, text_len=attention_mask.sum(dim=1).to(device), time=t,
            mask=mask, cond_mask=text_mask,
            return_ith_layer=model.config.repa_dit_layer, latent_cond=latent_cond,
        )
        pred = output["last_hidden_state"]
        tensors["py_pred_t0"] = pred.detach().cpu().numpy()
        
        # Unconditional
        x_uncond = x.clone()
        if not no_prompt:
            x_uncond[:, :latent_len] = 0
        
        neg_text = torch.zeros_like(text_condition)
        neg_text_len = attention_mask.sum(dim=1).to(device)
        tensors["neg_text"] = neg_text.detach().cpu().numpy()
        tensors["neg_text_len"] = neg_text_len.detach().cpu().numpy()
        
        null_output = model.transformer(
            x=x_uncond, text=neg_text, text_len=neg_text_len, time=t,
            mask=mask, cond_mask=text_mask,
            return_ith_layer=model.config.repa_dit_layer, latent_cond=empty_latent_cond,
        )
        null_pred = null_output["last_hidden_state"]
        tensors["py_null_pred_t0"] = null_pred.detach().cpu().numpy()
        
        if args.guidance_method == "cfg":
            vel = pred + (pred - null_pred) * args.guidance_strength
            tensors["py_velocity"] = vel.detach().cpu().numpy()
            return vel
        # APG ... (skipped for now or add if needed)
        return pred

    velocity_zero = evaluate_velocity(t_zero, y0.clone())
    tensors["velocity_zero"] = velocity_zero.detach().cpu().numpy()

    # 5. Full inference to get output latent
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_audio=load_audio(args.prompt_audio, sr) if not no_prompt else None,
        duration=duration,
        steps=args.nfe,
        cfg_strength=args.guidance_strength,
        guidance_method=args.guidance_method,
    )
    tensors["output_latent"] = output.latent.detach().cpu().numpy()
    tensors["output_waveform"] = output.waveform.detach().cpu().numpy()

    # Save all using safetensors
    from safetensors.torch import save_file
    # Convert all to torch tensors first if they are not
    save_dict = {}
    for k, v in tensors.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = torch.from_numpy(v).contiguous()
        else:
            save_dict[k] = torch.tensor(v).contiguous()
    
    save_file(save_dict, dump_dir / "tensors.safetensors")
    print(f"Dumped tensors to {dump_dir / 'tensors.safetensors'}")


if __name__ == "__main__":
    main()
