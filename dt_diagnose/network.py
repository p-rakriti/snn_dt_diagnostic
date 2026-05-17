"""LIF neuron with surrogate gradient and a feedforward SNN."""

import numpy as np
import torch
import torch.nn as nn


class LeakySurrogate(nn.Module):
    """Leaky integrate-and-fire neuron with arctan surrogate gradient.

    Standard SNN formulation: V[k] = beta * V[k-1] + input - spike-and-reset.
    Beta is exposed as a mutable attribute so the same network can be
    re-used at different dt values by setting net.lif1.beta = new_beta.
    """

    def __init__(self, beta, threshold=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.ATan.apply

    def forward(self, input_, mem):
        mem = self.beta * mem + input_
        mem_before_reset = mem.clone()
        spk = self.spike_gradient(mem - self.threshold)
        mem = mem * (1 - spk.detach())
        return spk, mem, mem_before_reset

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()
            ctx.save_for_backward(mem)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output
            return grad


class Net(nn.Module):
    """Feedforward SNN: input -> Linear -> LIF -> Linear -> LIF.

    Architecture sizes are passed in so the same class works for any
    dataset. Beta is computed from dt and tau_m at construction time
    but can be updated per-layer afterwards for cross-dt evaluation.
    """

    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = LeakySurrogate(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = LeakySurrogate(beta=beta)

    def forward(self, x, num_steps):
        """x shape: (num_steps, batch, num_inputs)."""
        B = x.shape[1]
        mem1 = torch.zeros((B, self.num_hidden), device=x.device)
        mem2 = torch.zeros((B, self.num_outputs), device=x.device)
        spk1_rec, spk2_rec, mem1_rec, mem2_rec = [], [], [], []
        for step in range(num_steps):
            cur1 = self.fc1(x[step].view(B, -1))
            spk1, mem1, mbr1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
            mem1_rec.append(mbr1)
            cur2 = self.fc2(spk1)
            spk2, mem2, mbr2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mbr2)
        return (torch.stack(spk1_rec, dim=0),
                torch.stack(spk2_rec, dim=0),
                torch.stack(mem1_rec, dim=0),
                torch.stack(mem2_rec, dim=0))


def set_beta(net, beta):
    """Update beta on all LIF layers in `net`. Used to re-target the same
    trained network at a different dt for evaluation."""
    for module in net.modules():
        if isinstance(module, LeakySurrogate):
            module.beta = beta
