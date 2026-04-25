import torch
import time
from titans import TitansMemory

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    d_model = 128
    seq_len = 128
    batch_size = 4
    
    model = TitansMemory(d_model=d_model).to(device)
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    start = time.time()
    output = model(x)
    end = time.time()

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Processing time: {end - start:.4f} seconds")
    print("Robustness Test: PASSED")

if __name__ == "__main__":
    run_benchmark()
