import numpy as np
import json
from pathlib import Path
from safetensors.numpy import load_file

def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs.reshape(-1).astype(np.float64)
    rhs = rhs.reshape(-1).astype(np.float64)
    lhs_norm = np.linalg.norm(lhs)
    rhs_norm = np.linalg.norm(rhs)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 1.0 if lhs_norm == rhs_norm else 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))

def compare_y0():
    py_path = Path("output/debug_py_new/tensors.safetensors")
    rs_path = Path("output/debug_rs_new")
    
    # Load Python y0
    py_tensors = load_file(str(py_path))
    py_y0 = py_tensors["y0"]
    
    # Load Rust y0
    manifest = json.loads((rs_path / "manifest.json").read_text())
    y0_entry = manifest["tensors"]["y0"]
    rs_y0 = np.fromfile(rs_path / y0_entry["file"], dtype=np.float32)
    rs_y0 = rs_y0.reshape(tuple(y0_entry["shape"]))
    
    print(f"Python y0 shape: {py_y0.shape}, mean: {py_y0.mean():.6f}, std: {py_y0.std():.6f}")
    print(f"Rust y0 shape: {rs_y0.shape}, mean: {rs_y0.mean():.6f}, std: {rs_y0.std():.6f}")
    
    if py_y0.shape != rs_y0.shape:
        print(f"Shape mismatch! Py: {py_y0.shape}, Rs: {rs_y0.shape}")
        return

    cos = cosine_similarity(py_y0, rs_y0)
    max_diff = np.max(np.abs(py_y0 - rs_y0))
    mean_diff = np.mean(np.abs(py_y0 - rs_y0))
    
    print(f"Cosine similarity: {cos:.10f}")
    print(f"Max absolute diff: {max_diff:.10f}")
    print(f"Mean absolute diff: {mean_diff:.10f}")

    if cos > 0.999999:
        print("\nSUCCESS: y0 is aligned!")
    else:
        print("\nFAILURE: y0 is NOT aligned.")

if __name__ == "__main__":
    compare_y0()
