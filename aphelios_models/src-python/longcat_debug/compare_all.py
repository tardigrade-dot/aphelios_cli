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

def compare_all():
    py_path = Path("output/debug_py_new/tensors.safetensors")
    rs_path = Path("output/debug_rs_new")
    
    # Load Python tensors
    py_tensors = load_file(str(py_path))
    
    # Load Rust tensors
    manifest = json.loads((rs_path / "manifest.json").read_text())
    
    mapping = {
        "text_condition": "text_condition",
        "y0": "y0",
        "velocity_zero": "velocity_zero",
        "output_latent": "output_latent",
        "output_waveform": "output_waveform",
        "latent_cond": "latent_cond",
        "prompt_latent": "prompt_latent",
        "py_pred_t0": "transformer_out_t0",
        "py_null_pred_t0": "null_pred_t0",
        "py_velocity": "velocity_zero", # Mapping velocity to check
    }
    
    results = []
    keys_to_check = list(mapping.keys())
    for key in py_tensors:
        if key not in keys_to_check:
            keys_to_check.append(key)

    for py_key in keys_to_check:
        if py_key not in py_tensors:
            continue
            
        rs_key = mapping.get(py_key, py_key)
        if rs_key not in manifest["tensors"]:
            continue
                
        py_val = py_tensors[py_key]
        entry = manifest["tensors"][rs_key]
        rs_val = np.fromfile(rs_path / entry["file"], dtype=np.float32)
        if len(entry["shape"]) > 0:
            rs_val = rs_val.reshape(tuple(entry["shape"]))
        else:
            rs_val = rs_val.reshape(1)

        if py_val.shape != rs_val.shape:
             if py_val.squeeze().shape == rs_val.squeeze().shape:
                 py_val = py_val.squeeze()
                 rs_val = rs_val.squeeze()
             else:
                 results.append(f"{py_key:20s}: SHAPE MISMATCH {py_val.shape} vs {rs_val.shape}")
                 continue

        cos = cosine_similarity(py_val, rs_val)
        max_diff = np.max(np.abs(py_val.astype(np.float64) - rs_val.astype(np.float64)))
        
        results.append(f"{py_key:20s}: cos={cos:.6f}, max_diff={max_diff:.10f}")
        
    print("\n".join(results))

if __name__ == "__main__":
    compare_all()
