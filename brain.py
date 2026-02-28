from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class BrainBatch:
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    W3: torch.Tensor
    b3: torch.Tensor

def init_brains(N: int, in_dim: int, h1: int, h2: int, out_dim: int, device: torch.device, dtype=torch.float32):
    def rand(shape, scale):
        return (torch.randn(shape, device=device, dtype=dtype) * scale)
    W1 = rand((N, in_dim, h1), (1.0 / np.sqrt(in_dim)))
    b1 = torch.zeros((N, h1), device=device, dtype=dtype)
    W2 = rand((N, h1, h2), (1.0 / np.sqrt(h1)))
    b2 = torch.zeros((N, h2), device=device, dtype=dtype)
    W3 = rand((N, h2, out_dim), (1.0 / np.sqrt(h2)))
    b3 = torch.zeros((N, out_dim), device=device, dtype=dtype)
    return BrainBatch(W1, b1, W2, b2, W3, b3)

def forward(brain: BrainBatch, x: torch.Tensor):
    h1 = torch.tanh(torch.bmm(x.unsqueeze(1), brain.W1).squeeze(1) + brain.b1)
    h2 = torch.tanh(torch.bmm(h1.unsqueeze(1), brain.W2).squeeze(1) + brain.b2)
    y = torch.bmm(h2.unsqueeze(1), brain.W3).squeeze(1) + brain.b3
    return h2, y

def mutate_brain(brain: BrainBatch, idxs: torch.Tensor, std: float):
    if idxs.numel() == 0:
        return
    def noise(t):
        return torch.randn_like(t[idxs]) * std
    brain.W1[idxs] += noise(brain.W1)
    brain.b1[idxs] += noise(brain.b1)
    brain.W2[idxs] += noise(brain.W2)
    brain.b2[idxs] += noise(brain.b2)
    brain.W3[idxs] += noise(brain.W3)
    brain.b3[idxs] += noise(brain.b3)

def copy_brain(src: BrainBatch, dst: BrainBatch, src_idx: torch.Tensor, dst_idx: torch.Tensor):
    dst.W1[dst_idx] = src.W1[src_idx]
    dst.b1[dst_idx] = src.b1[src_idx]
    dst.W2[dst_idx] = src.W2[src_idx]
    dst.b2[dst_idx] = src.b2[src_idx]
    dst.W3[dst_idx] = src.W3[src_idx]
    dst.b3[dst_idx] = src.b3[src_idx]

def baldwin_update_last_layer(brain: BrainBatch, h2: torch.Tensor, logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, lr: float):
    probs = torch.softmax(logits, dim=1)
    N = probs.shape[0]
    oh = torch.zeros_like(probs)
    oh[torch.arange(N, device=probs.device), actions] = 1.0
    adv = rewards - rewards.mean()
    dlogits = -(adv.unsqueeze(1) * (oh - probs))  # (N,3)

    out_dim = brain.b3.shape[1]
    dY = torch.zeros((N, out_dim), device=probs.device, dtype=probs.dtype)
    dY[:, 2:5] = dlogits  # mode logits live in last 3 outputs

    gradW3 = h2.unsqueeze(2) * dY.unsqueeze(1)
    gradb3 = dY

    brain.W3 -= lr * gradW3
    brain.b3 -= lr * gradb3
