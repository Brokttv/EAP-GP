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

An earlier version of this file called the toy network directly on the leaf
tensor (`G(gamma_leaf)`), never exercising an actual hook-shaped function that
receives the model's own activation and must RETURN a substitute. That gap let
a real bug through: attribute.py's shared hook did `point.clone();
new_input.requires_grad = True` unconditionally, which is illegal once `point`
already requires grad (a clone of a grad-requiring leaf is a non-leaf, and
PyTorch forbids setting .requires_grad on non-leaves) -- exactly Step A's call
site. It surfaced on the first real H100 run against GPT-2, not here, because
this test didn't route through a hook shape at all. Section 3 below closes
that gap by exercising both hook call sites explicitly.
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

# --- Section 3: exercise the actual hook shape used against a real HookedTransformer ---
# A real hook receives the model's own activation tensor and must return a substitute;
# it can't just call G(leaf) directly. That return-a-substitute shape is exactly where
# attribute.py's leaf/non-leaf bug lived, so replicate it here instead of bypassing it.

def make_grad_leaf_input_hook(leaf: torch.Tensor):
    def hook_fn(placeholder_activation):
        return leaf.clone()
    return hook_fn

def make_detached_input_hook(point: torch.Tensor):
    def hook_fn(placeholder_activation):
        return point.clone().requires_grad_(True)
    return hook_fn

placeholder = torch.zeros(batch, pos, d_model)  # stands in for the real activation tensor

# Step A's call site: the substituted tensor already requires grad.
grad_leaf = x_clean.clone().detach().requires_grad_(True)
hooked_input = make_grad_leaf_input_hook(grad_leaf)(placeholder)
out = G(hooked_input)
objective = (out - G_corrupted).pow(2).sum()
grad, = torch.autograd.grad(objective, grad_leaf)
assert torch.isfinite(grad).all(), "Step A hook: gradient did not flow back to the leaf"

# Step B's call site: the substituted tensor is a detached snapshot with no grad history.
detached_point = x_clean.clone().detach()
hooked_input_b = make_detached_input_hook(detached_point)(placeholder)
assert hooked_input_b.requires_grad, "Step B hook: substitute should require grad"
out_b = G(hooked_input_b)
out_b.sum().backward()
assert hooked_input_b.grad is not None and torch.isfinite(hooked_input_b.grad).all()

print("OK: hook-shaped substitution works for both the grad-requiring-leaf call site "
      "(Step A) and the detached-point call site (Step B).")
