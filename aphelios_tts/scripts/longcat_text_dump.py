import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
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
    parser = argparse.ArgumentParser(description="Offline LongCat text-condition dump")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--repo_dir", type=str, default="/Users/larry/coderesp/LongCat-AudioDiT")
    parser.add_argument("--tokenizer_dir", type=str, default="/Volumes/sw/pretrained_models/umt5-base")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    sys.path.insert(0, str(repo_dir))

    from audiodit import AudioDiTModel
    from utils import normalize_text

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = AudioDiTModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    text = normalize_text(args.text)
    if args.prompt_text:
        prompt_text = normalize_text(args.prompt_text)
        full_text = f"{prompt_text} {text}"
    else:
        full_text = text

    input_ids, attention_mask = batch_tokenize(args.tokenizer_dir, [full_text], pad_id=0)
    with torch.no_grad():
        text_condition = model.encode_text(input_ids.to(device), attention_mask.to(device))

    save_file(
        {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "lengths": attention_mask.sum(dim=1).cpu(),
            "text_condition": text_condition.cpu(),
        },
        args.dump_file,
    )
    print(f"Saved text dump to {args.dump_file}")


if __name__ == "__main__":
    main()
