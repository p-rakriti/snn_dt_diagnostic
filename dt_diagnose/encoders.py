"""Spike-train encoders for the three demo datasets.

Each encoder takes raw data + a chosen dt and returns a binary tensor
of shape (num_steps, batch, num_channels). The diagnostic eval calls
the appropriate encoder at dt_train and dt_test.
"""

import numpy as np
import torch


# ============================================================
# Latency MNIST (Zenke & Vogels 2021)
# ============================================================

def latency_encode(images, num_steps, dt, tau_enc=20.0, thr=0.2, tmax=None):
    """ZV latency encoding: each pixel x_i with x_i > thr emits one spike at
    t_i = tau_enc * log(x_i / (x_i - thr)). Pixels at or below thr don't spike.

    images: (B, 1, 28, 28) or (B, 28, 28) in [0, 1].
    Returns: (num_steps, B, 784) binary float tensor.
    """
    if tmax is None:
        tmax = num_steps * dt

    eps = 1e-7
    B = images.shape[0]
    P = 784
    x = images.view(B, -1)

    x_clipped = torch.clamp(x, min=thr + eps, max=1.0)
    t = tau_enc * torch.log(x_clipped / (x_clipped - thr))  # in ms

    no_spike = (x <= thr) | (t >= tmax)
    bin_idx = (t / dt).long().clamp(0, num_steps - 1)

    spikes = torch.zeros(num_steps, B, P, dtype=torch.float)
    valid = ~no_spike
    if valid.any():
        b_idx, p_idx = valid.nonzero(as_tuple=True)
        spikes[bin_idx[b_idx, p_idx], b_idx, p_idx] = 1.0
    return spikes


# ============================================================
# Rate-coded MNIST (Poisson)
# ============================================================

def rate_encode(images, num_steps, dt, input_rate_hz=50.0):
    """Generate Poisson spikes at the given dt from per-pixel intensities.

    For each pixel and each time bin, draws Bernoulli(p) where p is
    pixel intensity * (input_rate_hz * dt_seconds). Average per-pixel
    firing rate at maximum intensity equals input_rate_hz.

    images: (B, 1, 28, 28) or (B, 28, 28) in [0, 1].
    Returns: (num_steps, B, 784) binary float tensor.
    """
    gain = input_rate_hz * dt * 1e-3
    B = images.shape[0]
    rates = images.view(B, -1).unsqueeze(0).expand(num_steps, -1, -1) * gain
    rates = torch.clamp(rates, 0.0, 1.0)
    return torch.bernoulli(rates)


def rate_encode_padded(images, num_steps_train, dt_train, num_steps_test,
                       dt_test, input_rate_hz=50.0):
    """Generate Poisson spikes at dt_train, then zero-pad to dt_test.

    Preserves spike times in physical units across the two resolutions —
    this is the MNIST Fig 1 padded forward pass. The factor between dts
    must be integer (dt_train / dt_test).

    images: (B, 1, 28, 28) in [0, 1].
    Returns: tuple of two tensors
        spikes_train: (num_steps_train, B, 784) — at dt_train resolution
        spikes_test:  (num_steps_test,  B, 784) — same spike times,
                       zero-padded to dt_test resolution
    """
    assert dt_train > dt_test, "padded eval requires dt_train > dt_test"
    factor = round(dt_train / dt_test)
    assert abs(factor * dt_test - dt_train) < 1e-9, \
        f"dt_train / dt_test must be integer; got {dt_train}/{dt_test}"
    assert num_steps_test == num_steps_train * factor, \
        "num_steps_test must equal num_steps_train * (dt_train / dt_test)"

    spikes_train = rate_encode(images, num_steps_train, dt_train,
                                input_rate_hz=input_rate_hz)
    # Zero-pad: every factor-th bin in the fine-resolution stream gets a
    # spike from the coarse stream; the bins in between stay zero.
    B = spikes_train.shape[1]
    P = spikes_train.shape[2]
    spikes_test = torch.zeros(num_steps_test, B, P, dtype=torch.float)
    spikes_test[::factor] = spikes_train
    return spikes_train, spikes_test


# ============================================================
# SHD event binning
# ============================================================

def shd_bin_events(times, units, num_steps, dt, num_channels=700):
    """Bin a list of (time, channel) events into a binary tensor.

    times: array of spike times in seconds (the SHD HDF5 stores them in s)
    units: array of channel indices (same length as times)
    Returns: (num_steps, num_channels) binary float tensor for one sample.
    """
    dt_s = dt * 1e-3
    spikes = torch.zeros(num_steps, num_channels, dtype=torch.float)
    bin_idx = (times / dt_s).astype(np.int64)
    mask = (bin_idx >= 0) & (bin_idx < num_steps) & \
           (units >= 0) & (units < num_channels)
    if mask.any():
        spikes[bin_idx[mask], units[mask]] = 1.0
    return spikes


def shd_make_collate(num_steps, dt, num_channels=700):
    """Return a DataLoader collate_fn that bins SHD events at the given dt.

    Each item in the batch is a tuple (times, units, label) as returned by
    the SHDDataset class (see demo.py for the dataset definition).
    Returns a function that produces (spike_data, labels) where
        spike_data: (num_steps, batch, num_channels)
        labels:     (batch,) long tensor
    """
    def collate(batch):
        B = len(batch)
        spike_data = torch.zeros(num_steps, B, num_channels, dtype=torch.float)
        labels = torch.zeros(B, dtype=torch.long)
        for i, (times, units, label) in enumerate(batch):
            spike_data[:, i] = shd_bin_events(times, units, num_steps,
                                               dt, num_channels=num_channels)
            labels[i] = label
        return spike_data, labels
    return collate
