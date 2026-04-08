import torch
from pathlib import Path
from safetensors.numpy import load_file
import numpy as np

py_tensors = load_file("output/debug_py_new/tensors.safetensors")
rs_path = Path("output/debug_rs_new")
import json
manifest = json.loads((rs_path / "manifest.json").read_text())

# Need to dump modulation in Python.
# I'll add modulation dump in longcat_debug.py.
