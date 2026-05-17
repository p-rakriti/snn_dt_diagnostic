"""Bias rescaling utilities for cross-dt SNN evaluation.

The standard SNN formulation V[k] = beta*V[k-1] + W*X[k] + B applies
impulse-invariant discretisation (IID) to every term. IID is correct for
the spike input X[k] (which is impulse-like) but incorrect for the
constant bias B, which should be treated under zero-order-hold (ZOH).

A network trained at dt_train under IID implicitly encodes a continuous-time
bias B_bar = B_train / (1 - beta_train). Preserving B_bar at dt_test under
proper ZOH treatment requires
    B_test = (1 - beta_test) * B_bar
           = B_train * (1 - beta_test) / (1 - beta_train)
giving the rescaling factor implemented in `rho_bias_zoh`.

Weights are not rescaled because IID handles impulse inputs correctly.
"""

import math
import torch
import torch.nn as nn


def rho_bias_zoh(dt_train, dt_test, tau_m):
    """Closed-form ZOH bias rescaling factor.

    All three arguments in milliseconds (or any consistent time unit).
    Returns the multiplier such that B_test = rho * B_train preserves
    the steady-state membrane bias contribution under change of dt.

    Examples
    --------
    >>> rho_bias_zoh(dt_train=10.0, dt_test=1.0, tau_m=10.0)
    0.15048...   # biases shrink ~6.6x at fine dt
    >>> rho_bias_zoh(dt_train=10.0, dt_test=10.0, tau_m=10.0)
    1.0          # matched dt -> no rescaling
    """
    beta_train = math.exp(-dt_train / tau_m)
    beta_test = math.exp(-dt_test / tau_m)
    return (1.0 - beta_test) / (1.0 - beta_train)


def rescale_biases(net, rho):
    """Multiply every Linear layer's bias by `rho` in place.

    Layers without a bias (bias=False at construction) are skipped silently.
    Mutates `net` and returns it for chaining.
    """
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.mul_(float(rho))
    return net
