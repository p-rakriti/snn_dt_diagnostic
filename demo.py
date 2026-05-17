"""dt_diagnose demo: cross-resolution failure decomposition on three datasets.

For each of rate-MNIST, latency-MNIST, and SHD, this script:
  1. Loads a network trained at dt_train = 10 ms.
  2. Evaluates at matched dt_train (the ceiling).
  3. Evaluates at dt_test = 1 ms under three bias conditions:
        no_rescale  — biases untouched
        zoh_rescale — biases multiplied by closed-form ZOH factor
        zero_bias   — biases zeroed (upper bound on bias correction)
  4. Decomposes the cross-dt accuracy gap into bias-accumulation
     (closed-form rescuable) and coincidence-loss (not rescuable)
     components.

Accuracy is read from time-averaged output membrane potential, not
spike counts. Output spikes are sparse under cross-resolution eval;
membrane readout reveals the residual discriminative capacity.

Run:
    python demo.py

Output:
    Three-row table printed to terminal + results.png saved to repo root.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from dt_diagnose import (
    Net,
    latency_encode, rate_encode_padded, shd_make_collate,
    three_condition_eval, matched_dt_eval, failure_decomposition,
)


# ============================================================
# ===== Config ===============================================
# ============================================================

DT_TRAIN = 10.0          # ms — all three networks were trained at this dt
DT_TEST = 1.0            # ms — cross-resolution evaluation dt
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CHECKPOINTS_DIR = Path("checkpoints")
SHD_PATH = Path.home() / "data" / "hdspikes" / "shd_test.h5"
MNIST_PATH = "/tmp/data/mnist"

# Eval cap: 2048 images = 16 batches at batch_size=128. Quick but stable.
EVAL_MAX_BATCHES = 16
BATCH_SIZE = 128


# ============================================================
# ===== Dataset loaders ======================================
# ============================================================

def get_mnist_test_loader():
    """Standard torchvision MNIST test set, capped at EVAL_MAX_BATCHES."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    mnist_test = datasets.MNIST(MNIST_PATH, train=False, download=True,
                                 transform=transform)
    # Fixed subset for reproducibility
    indices = torch.arange(EVAL_MAX_BATCHES * BATCH_SIZE)
    subset = Subset(mnist_test, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                       drop_last=True)


class SHDTestDataset(Dataset):
    """Minimal SHD test-set wrapper: returns (times_in_seconds, units, label)."""
    def __init__(self, h5_path):
        self.f = h5py.File(h5_path, "r")
        self.times = self.f["spikes"]["times"]
        self.units = self.f["spikes"]["units"]
        self.labels = np.array(self.f["labels"], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (np.array(self.times[idx], dtype=np.float64),
                np.array(self.units[idx], dtype=np.int64),
                int(self.labels[idx]))


def get_shd_test_loader(dt_ms, T_ms=1000):
    """SHD test loader that bins events at the given dt."""
    num_steps = int(T_ms / dt_ms)
    ds = SHDTestDataset(SHD_PATH)
    collate = shd_make_collate(num_steps, dt_ms, num_channels=700)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                       drop_last=True, collate_fn=collate)


# ============================================================
# ===== forward_fn adapters (one per dataset) ================
# ============================================================
# Each takes (net, batch, dt, num_steps, device) and returns (mem2, targets).
# `num_steps` is unused here — we compute it from dt and T inside each adapter.
# The signature matches what three_condition_eval expects.

def make_latency_forward(T_ms=100, tau_enc=20.0, thr=0.2):
    def forward_fn(net, batch, dt, num_steps, device):
        images, targets = batch
        num_steps_dt = int(T_ms / dt)
        spike_data = latency_encode(images, num_steps_dt, dt,
                                     tau_enc=tau_enc, thr=thr,
                                     tmax=T_ms).to(device)
        targets = targets.to(device)
        _, _, _, mem2 = net(spike_data, num_steps_dt)
        return mem2, targets
    return forward_fn


def make_rate_padded_forward(T_ms=100, input_rate_hz=50.0):
    """For rate-MNIST: generate Poisson at dt_train, then zero-pad to dt
    if dt < dt_train. This preserves spike times across resolutions and
    isolates the bias-accumulation mechanism (no resampling-induced
    densification)."""
    def forward_fn(net, batch, dt, num_steps, device):
        images, targets = batch
        num_steps_train = int(T_ms / DT_TRAIN)
        num_steps_dt = int(T_ms / dt)
        if abs(dt - DT_TRAIN) < 1e-9:
            # Matched dt: just generate at the training resolution.
            from dt_diagnose import rate_encode
            spike_data = rate_encode(images, num_steps_train, DT_TRAIN,
                                       input_rate_hz=input_rate_hz)
        else:
            # Cross-dt: generate at dt_train, then zero-pad to dt.
            _, spike_data = rate_encode_padded(
                images, num_steps_train, DT_TRAIN,
                num_steps_dt, dt, input_rate_hz=input_rate_hz,
            )
        spike_data = spike_data.to(device)
        targets = targets.to(device)
        _, _, _, mem2 = net(spike_data, num_steps_dt)
        return mem2, targets
    return forward_fn


def make_shd_forward():
    def forward_fn(net, batch, dt, num_steps, device):
        spike_data, targets = batch
        num_steps_dt = spike_data.shape[0]
        spike_data = spike_data.to(device)
        targets = targets.to(device)
        _, _, _, mem2 = net(spike_data, num_steps_dt)
        return mem2, targets
    return forward_fn


# ============================================================
# ===== Per-dataset runner ===================================
# ============================================================

def run_dataset(name, ckpt_path, get_loader_train_dt, get_loader_test_dt,
                forward_fn, tau_m):
    """Run matched-dt eval + three-condition eval + decomposition for one dataset.

    `get_loader_*_dt` are callables taking no args and returning a DataLoader
    set up at the corresponding dt (matters for SHD, where binning is in the
    collate; for MNIST the loader is dt-agnostic).
    """
    print(f"\n=== {name} ===")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]

    num_inputs = cfg.get("num_inputs", 784 if "mnist" in name.lower() else 700)
    num_hidden = cfg["num_hidden"]
    num_outputs = cfg.get("num_outputs", 10 if "mnist" in name.lower() else 20)

    beta_train = np.exp(-DT_TRAIN / tau_m)
    net = Net(num_inputs=num_inputs, num_hidden=num_hidden,
              num_outputs=num_outputs, beta=float(beta_train)).to(DEVICE)
    net.load_state_dict(payload["state_dict"])

    # Matched-dt baseline
    matched_loader = get_loader_train_dt()
    matched = matched_dt_eval(
        net, forward_fn, DT_TRAIN, tau_m,
        matched_loader, device=DEVICE, max_batches=EVAL_MAX_BATCHES,
    )
    print(f"  matched dt={DT_TRAIN}ms: {matched['accuracy']:.2f}%")

    # Three-condition cross-resolution eval
    test_loader = get_loader_test_dt()
    results = three_condition_eval(
        net, forward_fn, DT_TRAIN, DT_TEST, tau_m,
        test_loader, device=DEVICE, max_batches=EVAL_MAX_BATCHES,
    )
    for cond, vals in results.items():
        print(f"  {cond:>12}: {vals['accuracy']:6.2f}%")

    decomp = failure_decomposition(
        no_rescale_acc=results["no_rescale"]["accuracy"],
        zoh_acc=results["zoh_rescale"]["accuracy"],
        zero_bias_acc=results["zero_bias"]["accuracy"],
        matched_acc=matched["accuracy"],
    )
    print(f"  decomposition: bias_accumulation={decomp['bias_accumulation_pct']:.0f}%"
          f"  coincidence_loss={decomp['coincidence_loss_pct']:.0f}%")
    return {
        "name": name,
        "matched": matched,
        "three_condition": results,
        "decomposition": decomp,
    }


# ============================================================
# ===== Plot =================================================
# ============================================================

def make_plot(all_results, save_path="results.png"):
    """Bar chart: three datasets x four conditions, with matched-dt baseline."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    datasets_order = [r["name"] for r in all_results]
    conditions = ["no_rescale", "zoh_rescale", "zero_bias"]
    x = np.arange(len(datasets_order))
    width = 0.22

    colors = {"no_rescale": "#cc4444", "zoh_rescale": "#4488cc", "zero_bias": "#888888"}

    for i, cond in enumerate(conditions):
        vals = [r["three_condition"][cond]["accuracy"] for r in all_results]
        ax.bar(x + (i - 1) * width, vals, width, label=cond, color=colors[cond])

    # Matched-dt baseline as a dashed line per dataset
    for i, r in enumerate(all_results):
        ax.hlines(r["matched"]["accuracy"],
                   x[i] - 1.5 * width, x[i] + 1.5 * width,
                   colors="black", linestyles="dashed", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order)
    ax.set_ylabel("Test accuracy at dt_test=1ms (%)")
    ax.set_title(f"Cross-resolution eval: dt_train={DT_TRAIN}ms -> dt_test={DT_TEST}ms\n"
                 "(dashed = matched-dt baseline)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nSaved plot to {save_path}")


# ============================================================
# ===== Main =================================================
# ============================================================

def main():
    all_results = []

    # Rate MNIST (padded eval)
    rate_fwd = make_rate_padded_forward(T_ms=100, input_rate_hz=50.0)
    all_results.append(run_dataset(
        name="rate_MNIST",
        ckpt_path=CHECKPOINTS_DIR / "rate_mnist.pt",
        get_loader_train_dt=get_mnist_test_loader,
        get_loader_test_dt=get_mnist_test_loader,
        forward_fn=rate_fwd,
        tau_m=10.0,
    ))

    # Latency MNIST
    latency_fwd = make_latency_forward(T_ms=100, tau_enc=20.0, thr=0.2)
    all_results.append(run_dataset(
        name="latency_MNIST",
        ckpt_path=CHECKPOINTS_DIR / "latency_mnist.pt",
        get_loader_train_dt=get_mnist_test_loader,
        get_loader_test_dt=get_mnist_test_loader,
        forward_fn=latency_fwd,
        tau_m=10.0,
    ))

    # SHD
    shd_fwd = make_shd_forward()
    all_results.append(run_dataset(
        name="SHD",
        ckpt_path=CHECKPOINTS_DIR / "shd.pt",
        get_loader_train_dt=lambda: get_shd_test_loader(DT_TRAIN),
        get_loader_test_dt=lambda: get_shd_test_loader(DT_TEST),
        forward_fn=shd_fwd,
        tau_m=10.0,
    ))

    print("\n========== summary ==========")
    for r in all_results:
        d = r["decomposition"]
        print(f"  {r['name']:>16}: matched={r['matched']['accuracy']:.0f}%  "
              f"no_rescale={r['three_condition']['no_rescale']['accuracy']:.0f}%  "
              f"zoh={r['three_condition']['zoh_rescale']['accuracy']:.0f}%  "
              f"| bias_accum={d['bias_accumulation_pct']:.0f}%  "
              f"coincidence={d['coincidence_loss_pct']:.0f}%")

    make_plot(all_results)


if __name__ == "__main__":
    main()
