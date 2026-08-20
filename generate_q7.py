import statistics
import time

from matplotlib import pyplot as plt

from src.evaluate import model_from_checkpoint
from pathlib import Path
import torch
import matplotlib

ROOT = Path.cwd()
checkpoint = (ROOT/ "afrilink_results/baseline/baseline_5000/checkpoints"/ "baseline_5000_step_005000.pt")

device = torch.device("cuda" if torch.cuda.is_available()else "mps" if torch.backends.mps.is_available()else "cpu")

model, _ = model_from_checkpoint(checkpoint, device)
prompt = torch.tensor([[257]], dtype=torch.long, device=device)

def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

@torch.no_grad()
def benchmark(tokens, use_cache):
    model.reset_cache()
    token_ids = prompt.clone()

    synchronize()
    start = time.perf_counter()

    if use_cache:
        logits = model(token_ids, use_cache=True)

        for step in range(tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            if step < tokens - 1:
                logits = model(next_token, use_cache=True)
    else:
        for _ in range(tokens):
            logits = model(token_ids, use_cache=False)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)

    synchronize()
    return time.perf_counter() - start

lengths = [16, 32, 64, 128, 256]
cached_throughput = []
uncached_throughput = []

# Warmup
benchmark(8, True)
benchmark(8, False)

for length in lengths:
    cached_times = [benchmark(length, True) for _ in range(3)]
    uncached_times = [benchmark(length, False) for _ in range(3)]

    cached_throughput.append(length / statistics.median(cached_times))
    uncached_throughput.append(length / statistics.median(uncached_times))

plt.figure(figsize=(7, 4))
plt.plot(lengths, cached_throughput, marker="o", label="KV cache")
plt.plot(lengths, uncached_throughput, marker="o", label="No cache")
plt.xlabel("Tokens generated")
plt.ylabel("Generation throughput (tokens/second)")
plt.title("Generation throughput with and without KV cache")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("q7_generation_throughput.png", dpi=200)
plt.show()

speedup = cached_throughput[-1] / uncached_throughput[-1]
print("Speedup at 256 tokens:", speedup)