import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


DEFAULT_TEXTS = [
    "在二十世纪上半叶的中国乡村，有两个巨大的历史进程值得注意，它们使此一时期的中国有别于前一时代。",
    "贝克莱的努力并未产生有形的结果，但它促使后来的研究者重新思考知识与经验之间的关系。",
    "请在系统启动后，先检查网络配置，再依次验证日志写入、配置加载和模型初始化是否正常。",
    "This is a short bilingual sample for alignment testing between Rust and Python implementations.",
    "如果我们把随机噪声、文本编码和提示音频编码逐项固定下来，就能更准确地定位音质下降发生在哪个阶段。",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    env = dict(**__import__("os").environ)
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    env.setdefault("HF_HUB_OFFLINE", "1")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def load_rust_dump(dump_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    tensors: dict[str, np.ndarray] = {}
    dtype_map = {
        "f32": np.float32,
        "u32": np.uint32,
        "i64": np.int64,
    }
    for name, entry in manifest["tensors"].items():
        dtype = np.dtype(dtype_map[entry["dtype"]])
        shape = tuple(entry["shape"])
        data = np.fromfile(dump_dir / entry["file"], dtype=dtype)
        tensors[name] = data.reshape(shape)
    return tensors, manifest["scalars"]


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs.reshape(-1).astype(np.float64)
    rhs = rhs.reshape(-1).astype(np.float64)
    lhs_norm = np.linalg.norm(lhs)
    rhs_norm = np.linalg.norm(rhs)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 1.0 if lhs_norm == rhs_norm else 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    return {
        "shape_match": list(lhs.shape) == list(rhs.shape),
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "max_abs_diff": float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)))),
        "mean_abs_diff": float(np.mean(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)))),
        "cosine": cosine_similarity(lhs, rhs),
        "lhs_mean": float(lhs.astype(np.float64).mean()),
        "rhs_mean": float(rhs.astype(np.float64).mean()),
        "lhs_std": float(lhs.astype(np.float64).std()),
        "rhs_std": float(rhs.astype(np.float64).std()),
    }


def read_samples(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_TEXTS
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch LongCat Rust/Python alignment comparison")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=Path("/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B"))
    parser.add_argument("--prompt_text", type=str, default="贝克莱的努力并未产生有形的结果")
    parser.add_argument("--prompt_audio", type=Path, default=Path("/Volumes/sw/video/youyi-5s.wav"))
    parser.add_argument("--python_bin", type=Path, default=Path("/Volumes/sw/conda_envs/lcataudio/bin/python"))
    parser.add_argument("--cargo_bin", type=str, default="cargo")
    parser.add_argument("--sample_file", type=Path, default=None)
    parser.add_argument("--guidance_method", type=str, default="cfg", choices=["cfg", "apg"])
    parser.add_argument("--guidance_strength", type=float, default=4.0)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cargo_features", type=str, default="metal")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.sample_file)

    report: dict[str, object] = {
        "samples": [],
        "aggregate": {},
    }

    rust_base_cmd = [args.cargo_bin, "run", "-p", "aphelios_tts", "--example", "longcat_debug_dump"]
    if args.release:
        rust_base_cmd.append("--release")
    if args.cargo_features:
        rust_base_cmd.extend(["--features", args.cargo_features])
    rust_base_cmd.append("--")
    if args.cpu:
        rust_base_cmd.append("--cpu")

    for idx, text in enumerate(samples, start=1):
        sample_dir = output_dir / f"sample_{idx:02d}"
        py_dir = sample_dir / "python"
        rust_native_dir = sample_dir / "rust_native"
        rust_replay_dir = sample_dir / "rust_replay"
        py_dir.mkdir(parents=True, exist_ok=True)

        py_cmd = [
            str(args.python_bin),
            str(repo_root / "aphelios_tts/scripts/longcat_debug.py"),
            "--text",
            text,
            "--prompt_text",
            args.prompt_text,
            "--prompt_audio",
            str(args.prompt_audio),
            "--dump_dir",
            str(py_dir),
            "--model_dir",
            str(args.model_dir),
            "--nfe",
            str(args.nfe),
            "--guidance_strength",
            str(args.guidance_strength),
            "--guidance_method",
            args.guidance_method,
            "--seed",
            str(args.seed),
        ]
        run(py_cmd, repo_root)

        common_rust_args = [
            "--text",
            text,
            "--prompt_text",
            args.prompt_text,
            "--prompt_audio",
            str(args.prompt_audio),
            "--dump_dir",
            str(rust_native_dir),
            "--model_dir",
            str(args.model_dir),
            "--nfe",
            str(args.nfe),
            "--guidance_strength",
            str(args.guidance_strength),
            "--guidance_method",
            args.guidance_method,
            "--seed",
            str(args.seed),
        ]
        run(rust_base_cmd + common_rust_args, repo_root)

        replay_args = common_rust_args.copy()
        replay_args[replay_args.index(str(rust_native_dir))] = str(rust_replay_dir)
        replay_args.extend(["--override_from", str(py_dir / "tensors.safetensors")])
        run(rust_base_cmd + replay_args, repo_root)

        py_tensors = load_file(py_dir / "tensors.safetensors")
        rust_native, rust_native_scalars = load_rust_dump(rust_native_dir)
        rust_replay, rust_replay_scalars = load_rust_dump(rust_replay_dir)

        native_metrics = {
            "input_ids": compare_arrays(rust_native["input_ids"], py_tensors["input_ids"]),
            "attention_mask": compare_arrays(rust_native["attention_mask"], py_tensors["attention_mask"]),
            "text_condition": compare_arrays(rust_native["text_condition"], py_tensors["text_condition"]),
            "prompt_latent": compare_arrays(rust_native["prompt_latent"], py_tensors["prompt_latent"]),
            "duration": {
                "rust": rust_native_scalars["duration"],
                "python": int(py_tensors["duration"].reshape(-1)[0]),
                "match": rust_native_scalars["duration"] == int(py_tensors["duration"].reshape(-1)[0]),
            },
        }
        replay_metrics = {
            "y0": compare_arrays(rust_replay["y0"], py_tensors["y0"]),
            "transformer_out_t0": compare_arrays(rust_replay["transformer_out_t0"], py_tensors["py_transformer_out"]),
            "velocity_zero": compare_arrays(rust_replay["velocity_zero"], py_tensors["velocity_zero"]),
            "output_latent": compare_arrays(rust_replay["output_latent"], py_tensors["output_latent"]),
            "output_waveform": compare_arrays(rust_replay["output_waveform"], py_tensors["output_waveform"]),
            "duration": {
                "rust": rust_replay_scalars["duration"],
                "python": int(py_tensors["duration"].reshape(-1)[0]),
                "match": rust_replay_scalars["duration"] == int(py_tensors["duration"].reshape(-1)[0]),
            },
        }

        sample_report = {
            "index": idx,
            "text": text,
            "native": native_metrics,
            "replay": replay_metrics,
        }
        report["samples"].append(sample_report)

        print(f"\n=== Sample {idx:02d} ===")
        print(text)
        print(
            "native text_condition cosine={:.6f}, mean_abs={:.6e}".format(
                native_metrics["text_condition"]["cosine"],
                native_metrics["text_condition"]["mean_abs_diff"],
            )
        )
        print(
            "native prompt_latent cosine={:.6f}, mean_abs={:.6e}".format(
                native_metrics["prompt_latent"]["cosine"],
                native_metrics["prompt_latent"]["mean_abs_diff"],
            )
        )
        print(
            "replay velocity_zero cosine={:.6f}, mean_abs={:.6e}".format(
                replay_metrics["velocity_zero"]["cosine"],
                replay_metrics["velocity_zero"]["mean_abs_diff"],
            )
        )
        print(
            "replay output_waveform cosine={:.6f}, mean_abs={:.6e}".format(
                replay_metrics["output_waveform"]["cosine"],
                replay_metrics["output_waveform"]["mean_abs_diff"],
            )
        )

    aggregate = {}
    for section, keys in {
        "native": ["text_condition", "prompt_latent"],
        "replay": ["transformer_out_t0", "velocity_zero", "output_latent", "output_waveform"],
    }.items():
        aggregate[section] = {}
        for key in keys:
            cosines = [sample[section][key]["cosine"] for sample in report["samples"]]
            mean_abs = [sample[section][key]["mean_abs_diff"] for sample in report["samples"]]
            aggregate[section][key] = {
                "avg_cosine": float(np.mean(cosines)),
                "avg_mean_abs_diff": float(np.mean(mean_abs)),
                "min_cosine": float(np.min(cosines)),
                "max_mean_abs_diff": float(np.max(mean_abs)),
            }
    report["aggregate"] = aggregate

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
