"""
Standalone sanity check of the EAP-GP GradPath autograd mechanic, isolated from
transformer_lens/HookedTransformer (not installed / no GPU in this sandbox).

This mimics the hook pattern used in src/eap/attribute.py:get_scores_eap_gp:
  - a "hooked" leaf tensor is substituted mid-forward-pass
  - we backprop an objective through the rest of the forward pass to get the
    gradient w.r.t. that leaf
  - we take a unit-normalized step against the gradient
  - we repeat for k steps and confirm the objective (dist to corrupted output)
    generally decreases and the loop produces finite, distinct path points.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

d_model = 8
batch, pos = 2, 3
steps = 5

# A toy "downstream network" standing in for everything after the input node
# (attention layers, MLPs, unembed) in a real transformer.
net = nn.Sequential(
    nn.Linear(d_model, 16), nn.GELU(),
    nn.Linear(16, 16), nn.GELU(),
    nn.Linear(16, d_model),
)

x_clean = torch.randn(batch, pos, d_model)
x_corrupted = torch.randn(batch, pos, d_model)

with torch.no_grad():
    G_corrupted = net(x_corrupted)  # G(x_u'), cached once, analogous to corrupted_logits

def G(point: torch.Tensor) -> torch.Tensor:
    return net(point)

# --- Step A: GradPath construction ---
gamma = x_clean.clone().detach()
path_points = []
objectives = []
for j in range(steps):
    gamma_leaf = gamma.clone().detach().requires_grad_(True)
    out = G(gamma_leaf)
    diff = out - G_corrupted
    objective = diff.pow(2).sum()
    objectives.append(objective.item())

    grad, = torch.autograd.grad(objective, gamma_leaf)
    step_norm = grad.norm(p=2).clamp_min(1e-12)
    gamma = gamma_leaf.detach() - grad / step_norm
    path_points.append(gamma.clone())

    assert torch.isfinite(gamma).all(), f"non-finite path point at step {j}"
    assert (gamma - gamma_leaf.detach()).norm() > 0, f"path point did not move at step {j}"

print("objective ||G(gamma_j) - G(x_u')||^2 across steps:", [round(o, 4) for o in objectives])
assert objectives[-1] < objectives[0], "GradPath objective should trend toward the corrupted output"

# distinct points (path shouldn't collapse to a single repeated vector)
for i in range(1, steps):
    assert not torch.allclose(path_points[i], path_points[i - 1]), f"path stalled at step {i}"

# --- Step B: attribution-style backward through a chosen path point ---
scores_accum = torch.zeros(d_model)
for point in path_points:
    leaf = point.clone().detach().requires_grad_(True)
    logits = G(leaf)
    metric_value = logits.sum()  # stand-in for the real `metric(...)`
    metric_value.backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    scores_accum += leaf.grad.sum(dim=(0, 1))

print("accumulated (toy) attribution scores:", scores_accum)
print("OK: GradPath autograd mechanic behaves as expected.")
