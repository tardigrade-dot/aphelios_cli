#! /Volumes/sw/conda_envs/onnx/bin/python

from pathlib import Path

from huggingface_hub import snapshot_download as hfd
from modelscope import snapshot_download as msd
import os

model_id = "AngelSlim/Hy-MT1.5-1.8B-1.25bit"
model_dir = "/Volumes/sw/pretrained_models"
source = "modelscope"# modelscope or huggingface

hf_token = os.environ.get('HF_TOKEN')
ms_token = os.environ.get("MODELSCOPE_TOKEN")

model_name = model_id.split("/")[1]

local_path = Path(model_dir).joinpath(model_name)

print(f"hf token {hf_token}")

match source:
    case "modelscope":
        local_path = msd(
            repo_id=model_id,
            local_dir=local_path,
            # token=ms_token,
            # local_dir_use_symlinks=False
        )
    case "huggingface":
        local_path = hfd(
            repo_id=model_id,
            local_dir=local_path,
            token=hfd,
            local_dir_use_symlinks=False
        )