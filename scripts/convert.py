#! /Volumes/sw/conda_envs/onnx/bin/python
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from onnxruntime_genai.models.builder import create_model
def main():
    parser = argparse.ArgumentParser()
    
    model_name = "Dolphin-v2"
    model_dir = "/Volumes/sw/pretrained_models"
    model_path = Path(model_dir).joinpath(model_name)

    work_dir = Path("/Users/larry/coderesp/aphelios_cli/output/").joinpath(model_name)
    onnx_dir = work_dir / "onnx"
    cache_dir = work_dir / "cache"

    work_dir.mkdir(exist_ok=True)
    model_path.mkdir(exist_ok=True)
    onnx_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    create_model(
        model_name=model_name,
        input_path=str(model_path),     # HF model directory
        output_dir=str(onnx_dir),     # ONNX output
        precision="fp16",              # fp32 | fp16 | int8 | int4 (if supported)
        execution_provider="cpu",      # cpu | cuda | dml | coreml
        cache_dir=str(work_dir / "cache"),  # optional cache
        extra_options={}
    )

    print("\n✅ Done")
    print(f"ONNX model at: {onnx_dir}")


if __name__ == "__main__":
    main()
