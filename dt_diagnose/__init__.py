"""dt_diagnose — cross-resolution failure-mode diagnostic for surrogate-gradient SNNs.

Public API:
    Net, LeakySurrogate, set_beta             (from .network)
    latency_encode, rate_encode,
        rate_encode_padded, shd_bin_events,
        shd_make_collate                       (from .encoders)
    rho_bias_zoh, rescale_biases               (from .rescaling)
    three_condition_eval, matched_dt_eval,
        failure_decomposition                  (from .eval)
"""

from .network import Net, LeakySurrogate, set_beta
from .encoders import (
    latency_encode,
    rate_encode,
    rate_encode_padded,
    shd_bin_events,
    shd_make_collate,
)
from .rescaling import rho_bias_zoh, rescale_biases
from .eval import three_condition_eval, matched_dt_eval, failure_decomposition

__all__ = [
    "Net", "LeakySurrogate", "set_beta",
    "latency_encode", "rate_encode", "rate_encode_padded",
    "shd_bin_events", "shd_make_collate",
    "rho_bias_zoh", "rescale_biases",
    "three_condition_eval", "matched_dt_eval", "failure_decomposition",
]
