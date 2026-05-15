"""
Extract minimal final-checkpoint payloads from the full training files
(which contain training histories, all checkpoints, gradient logs, etc).

Run once. Output: three small .pt files in checkpoints/ that the demo loads.
"""
import os
import torch

HOME = os.path.expanduser("~")

SOURCES = [
    {
        "src": f"{HOME}/Documents/ET26/rate_coded_mnist/rate_mnist_train_dt10_seed_6.pt",
        "dst": "checkpoints/rate_mnist.pt",
        "dataset": "rate_MNIST",
    },
{
        "src": f"{HOME}/Documents/ET26/latency_mnist/latency_mnist_fig1_zohbias_train_dt10_seed_6.pt",
        "dst": "checkpoints/latency_mnist.pt",
        "dataset": "latency_MNIST",
    },
    {
        "src": f"{HOME}/Documents/ET26/shd/shd_fig1_train_dt10_seed_6.pt",
        "dst": "checkpoints/shd.pt",
        "dataset": "SHD",
    },
]
for entry in SOURCES:
    print(f"Reading {entry['src']}")
    r = torch.load(entry["src"], map_location="cpu", weights_only=False)
    final_iter = max(r["checkpoints"].keys())
    payload = {
        "dataset": entry["dataset"],
        "state_dict": r["checkpoints"][final_iter],
        "config": r["config"],
        "dt_train": r.get("dt_train", r["config"].get("dt_train", None)),
    }
    torch.save(payload, entry["dst"])
    sz_kb = os.path.getsize(entry["dst"]) / 1024
    print(f"  -> {entry['dst']} ({sz_kb:.0f} KB)")

print("\ndone.")

