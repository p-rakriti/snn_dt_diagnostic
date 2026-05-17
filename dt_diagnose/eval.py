"""Three-condition evaluation and failure decomposition for cross-dt diagnostic.

The diagnostic runs each trained network at dt_test under three conditions:
    no_rescale  — biases untouched
    zoh_rescale — biases multiplied by rho_bias (closed-form ZOH fix)
    zero_bias   — biases set to zero (upper bound on any bias correction)

The matched-dt accuracy (network evaluated at its training dt) provides
the ceiling. The gap from matched-dt to no_rescale decomposes as:
    no_rescale -> zoh_rescale gap     = bias accumulation (bias accumulation,
                                          rescuable by ZOH)
    zoh_rescale -> matched-dt gap     = coincidence loss (coincidence loss,
                                          not closed-form rescuable)
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .network import set_beta
from .rescaling import rescale_biases


def _membrane_accuracy(mem2, targets):
    """Sum membrane potentials over time, take argmax over output units.
    Returns accuracy as a float in [0, 100]."""
    _, pred = mem2.sum(dim=0).max(dim=1)
    return float((pred == targets).float().mean().item()) * 100


def _ce_loss_over_time(mem2, targets, loss_fn):
    """Per-step cross-entropy summed across time (the per-step loss used
    during training). Returns a Python float."""
    total = 0.0
    for step in range(mem2.shape[0]):
        total += float(loss_fn(mem2[step], targets).item())
    return total


def three_condition_eval(
    net,
    forward_fn,
    dt_train,
    dt_test,
    tau_m,
    test_loader,
    device="cpu",
    max_batches=None,
):
    """Run no_rescale / zoh_rescale / zero_bias at dt_test on a trained net.

    Parameters
    ----------
    net : torch.nn.Module
        The trained network. Will be load_state_dict'd from a snapshot
        before each condition, so the caller's net object is not
        permanently modified.
    forward_fn : callable
        forward_fn(net, batch, dt, num_steps) -> (output_mem, targets)
        Takes care of input encoding (latency / rate-padded / SHD-binned)
        and returns the output-layer membrane potentials and the labels
        in the format the eval expects.
    dt_train, dt_test, tau_m : float
        In milliseconds.
    test_loader : iterable
        Yields (raw_batch, targets) tuples. The forward_fn is responsible
        for converting raw_batch to a spike tensor at the right dt.
    device : str
        Where to run the forward pass.
    max_batches : int or None
        Optional cap on batches per condition (for quick smoke-tests).

    Returns
    -------
    dict mapping condition name -> dict with keys "accuracy" and "loss".
    """
    from math import exp

    beta_test = exp(-dt_test / tau_m)
    rho = (1.0 - beta_test) / (1.0 - exp(-dt_train / tau_m))

    conditions = [
        ("no_rescale", 1.0),
        ("zoh_rescale", rho),
        ("zero_bias", 0.0),
    ]

    # Snapshot the trained weights so we can re-load before each condition.
    trained_state = copy.deepcopy(net.state_dict())
    loss_fn = nn.CrossEntropyLoss()

    results = {}
    for name, scale in conditions:
        # Reset to trained weights, then apply this condition's bias scaling.
        net.load_state_dict(trained_state)
        rescale_biases(net, scale)
        set_beta(net, beta_test)
        net.eval()

        accs, losses = [], []
        with torch.no_grad():
            for b, batch in enumerate(test_loader):
                if max_batches is not None and b >= max_batches:
                    break
                mem2, targets = forward_fn(net, batch, dt_test,
                                            num_steps=None,
                                            device=device)
                accs.append(_membrane_accuracy(mem2, targets))
                losses.append(_ce_loss_over_time(mem2, targets, loss_fn))

        results[name] = {
            "accuracy": float(np.mean(accs)),
            "loss": float(np.mean(losses)),
            "n_batches": len(accs),
        }

    # Restore the network to its original trained state for the caller.
    net.load_state_dict(trained_state)
    return results


def matched_dt_eval(
    net,
    forward_fn,
    dt_train,
    tau_m,
    test_loader,
    device="cpu",
    max_batches=None,
):
    """Evaluate the trained network at its training dt (no rescaling).

    Used to establish the matched-dt baseline (the ceiling for failure
    decomposition).
    """
    beta_train = math.exp(-dt_train / tau_m)
    trained_state = copy.deepcopy(net.state_dict())

    set_beta(net, beta_train)
    net.eval()
    loss_fn = nn.CrossEntropyLoss()

    accs, losses = [], []
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            if max_batches is not None and b >= max_batches:
                break
            mem2, targets = forward_fn(net, batch, dt_train,
                                        num_steps=None,
                                        device=device)
            accs.append(_membrane_accuracy(mem2, targets))
            losses.append(_ce_loss_over_time(mem2, targets, loss_fn))

    net.load_state_dict(trained_state)
    return {
        "accuracy": float(np.mean(accs)),
        "loss": float(np.mean(losses)),
        "n_batches": len(accs),
    }


def failure_decomposition(no_rescale_acc, zoh_acc, zero_bias_acc, matched_acc):
    """Decompose the cross-dt accuracy gap into bias accumulation and 2b fractions.

    Parameters
    ----------
    no_rescale_acc, zoh_acc, zero_bias_acc : float
        Accuracies under the three conditions at dt_test, in percentage.
    matched_acc : float
        Accuracy at dt_train (the ceiling), in percentage.

    Returns
    -------
    dict with keys:
        total_gap         — matched_acc - no_rescale_acc
        bias_accumulation_pct     — fraction of total_gap closed by ZOH rescaling
        coincidence_loss_pct     — fraction of total_gap that remains after ZOH
        zero_bias_ceiling — zero_bias_acc - matched_acc (negative if any
                             gap is closed beyond what ZOH did; positive
                             if zero_bias overshoots matched, which would
                             flag an unexpected interaction)
    """
    total_gap = matched_acc - no_rescale_acc
    bias_recovery = zoh_acc - no_rescale_acc
    residual = matched_acc - zoh_acc

    if total_gap <= 0:
        # No gap to decompose (matched performance is at or below no_rescale).
        return {
            "total_gap": total_gap,
            "bias_accumulation_pct": 0.0,
            "coincidence_loss_pct": 0.0,
            "zero_bias_ceiling": zero_bias_acc - matched_acc,
            "note": "no positive gap to decompose",
        }

    return {
        "total_gap": total_gap,
        "bias_accumulation_pct": 100.0 * bias_recovery / total_gap,
        "coincidence_loss_pct": 100.0 * residual / total_gap,
        "zero_bias_ceiling": zero_bias_acc - matched_acc,
    }
