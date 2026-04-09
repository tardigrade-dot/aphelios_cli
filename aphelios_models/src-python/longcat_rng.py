import numpy as np
import torch

def mix_seed(x: int) -> int:
    # 64-bit unsigned wrapping arithmetic
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)

class LongCatRng:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF
        self.cached_normal = None

    def fork(self, stream: int):
        return LongCatRng(mix_seed(self.state ^ (stream & 0xFFFFFFFFFFFFFFFF)))

    def next_u64(self) -> int:
        self.state = mix_seed(self.state)
        return self.state

    def next_unit_open_open(self) -> float:
        bits = self.next_u64() >> 40
        # This matches the Rust implementation:
        # ((bits as f32) + 0.5) / ((1u32 << 24) as f32)
        return (float(bits) + 0.5) / float(1 << 24)

    def next_standard_normal(self) -> float:
        if self.cached_normal is not None:
            val = self.cached_normal
            self.cached_normal = None
            return val
        
        # Box-Muller transform
        u1 = max(self.next_unit_open_open(), 1.17549435e-38) # f32::MIN_POSITIVE
        u2 = self.next_unit_open_open()
        
        radius = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        
        z0 = radius * np.cos(theta)
        z1 = radius * np.sin(theta)
        
        # Ensure we keep f32 precision to match Rust
        self.cached_normal = float(np.float32(z1))
        return float(np.float32(z0))

    def standard_normal_tensor(self, shape, dtype=torch.float32, device='cpu'):
        elem_count = 1
        for s in shape:
            elem_count *= s
            
        values = []
        for _ in range(elem_count):
            values.append(self.next_standard_normal())
            
        return torch.tensor(values, dtype=dtype, device=device).reshape(shape)

if __name__ == "__main__":
    # Simple test to match Rust's seeded_normal_stream_is_deterministic
    rng1 = LongCatRng(1024)
    t1 = rng1.standard_normal_tensor((2, 3))
    print("RNG(1024) [2, 3]:")
    print(t1)
    
    rng2 = LongCatRng(1024)
    t2 = rng2.standard_normal_tensor((2, 3))
    print("\nRNG(1024) [2, 3] (repeat):")
    print(t2)
    
    assert torch.allclose(t1, t2)
    print("\nDeterministic test passed!")
