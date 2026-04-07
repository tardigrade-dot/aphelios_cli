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
    parser = argparse.ArgumentParser(description="Offline LongCat-AudioDiT reference inference")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--prompt_audio", type=str, default=None)
    parser.add_argument("--output_audio", type=str, required=True)
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

    import audiodit  # noqa: F401
    from audiodit import AudioDiTModel
    from utils import approx_duration_from_text, load_audio, normalize_text

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(args.seed)

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

    if not no_prompt:
        prompt_wav = load_audio(args.prompt_audio, sr).unsqueeze(0)
        off = 3
        pw = load_audio(args.prompt_audio, sr)
        if pw.shape[-1] % full_hop != 0:
            pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        pw = F.pad(pw, (0, full_hop * off))
        with torch.no_grad():
            plt = model.vae.encode(pw.unsqueeze(0).to(device))
        if off:
            plt = plt[..., :-off]
        prompt_dur = plt.shape[-1]
    else:
        prompt_wav = None
        prompt_dur = 0

    prompt_time = prompt_dur * full_hop / sr
    dur_sec = approx_duration_from_text(text, max_duration=max_duration - prompt_time)
    if prompt_text is not None:
        approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
        ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
        dur_sec = dur_sec * ratio

    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_audio=prompt_wav,
        duration=duration,
        steps=args.nfe,
        cfg_strength=args.guidance_strength,
        guidance_method=args.guidance_method,
    )

    wav = output.waveform.squeeze().detach().cpu().numpy()
    sf.write(args.output_audio, wav, sr)


if __name__ == "__main__":
    main()
