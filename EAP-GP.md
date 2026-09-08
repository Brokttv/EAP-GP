# EAP-GP

An implementation of **EAP-GP** ("Edge Attribution Patching with GradPath"), from

> Zhang, Dong, Zhang, Yang, Hu, Liu, Zhou, Wang. *EAP-GP: Mitigating Saturation Effect in
> Gradient-based Automated Circuit Identification*. [arXiv:2502.06852](https://arxiv.org/abs/2502.06852) (2025).

The paper does not release code. This builds on Michael Hanna's
[EAP-IG](https://github.com/hannamw/EAP-IG) codebase (MIT licensed; see this repo's `LICENSE`)
by adding `get_scores_eap_gp` in [`src/eap/attribute.py`](src/eap/attribute.py), wired in as
`method='EAP-GP'`. Everything else (graph construction, evaluation, metrics, notebooks) is
unmodified upstream EAP-IG.

## What EAP-GP changes vs. EAP-IG

Both methods estimate every edge's attribution score with the same first-order formula

```
score(u, v) = (x_u - x_u') · mean_j[ ∂L(path(j)) / ∂x_v ]
```

where `x_u`/`x_u'` are node `u`'s clean/corrupted activations and the mean is taken over `k`
points along a path between them (this repo, like upstream EAP-IG's cheap `EAP-IG-inputs`
variant, applies the path only at the model's input activations and lets it propagate through
the rest of the network — that's what makes the method O(k) forward/backward passes instead of
O(k · n_layers)).

- **EAP-IG** uses the straight line `path(j) = x_u' + (j/k)(x_u - x_u')`.
- **EAP-GP** replaces that with a *GradPath*: starting at `path(0) = x_u` (clean), each step takes
  a unit-normalized gradient-descent step that pulls the path point toward matching the
  *corrupted* run's full output `G(x_u')`:

  ```
  g_(j+1) = ∇_path(j) || G(path(j)) - G(x_u') ||²₂
  path(j+1) = path(j) - g_(j+1) / ||g_(j+1)||₂
  ```

  The intuition (Section 5 of the paper): a fixed straight-line path can land in regions where
  `∂L/∂x_v ≈ 0` (saturation), silently zeroing out an edge's estimated importance even when the
  edge matters. Since GradPath actively moves toward the corrupted output, it's constructed to
  avoid exactly those flat regions.

This changes the *points* the gradient is averaged over; the outer `(x_u - x_u')` term and the
edge-score bookkeeping (via `make_hooks_and_matrices`'s forward/backward hooks) are untouched
from EAP-IG.

## Usage

```python
from eap.attribute import attribute

scores = attribute(
    model, graph, dataloader, metric,
    method='EAP-GP',
    ig_steps=5,   # k in the paper; they find k=5 works best (ablated over k in [3, 20])
)
```

This otherwise follows the same `Graph` / dataloader / metric conventions as upstream
`EAP-IG` — see `UPSTREAM_README.md` and `ioi.ipynb` / `greater_than.ipynb` for the surrounding
workflow (building a `Graph` from a `HookedTransformer`, running `attribute`, then
`graph.apply_topn(...)` and `evaluate_graph(...)`).

## Implementation notes / assumptions

The paper states the core update per-node (`x_u` any upstream node's activation, `G(s) = G(E(z) - x_u + s)`
the model's output with just that node patched). Doing that literally, per node, would require its
own k-step gradient descent for *every* node in the graph — intractable, and not what the paper's
reported ~5x-slower-than-EAP-IG number is consistent with (a literal per-node GradPath would be
orders of magnitude slower, not 5x). Upstream EAP-IG resolves the same tension for its cheap
variant by building a *single* joint path at the model's input and reusing the standard
first-order-Taylor edge bookkeeping for every downstream node from one pair of forward/backward
passes per step. This implementation applies the same trick to GradPath: one GradPath is built
over the input activations (Step A, `steps` extra forward+backward passes), then reused for every
edge's attribution (Step B, `steps` more forward+backward passes) — which is what produces the
observed-in-the-paper ~2x-plus-overhead, matching their reported ~5x wall-clock cost relative to
EAP-IG on GPT-2 small / IOI.

Two judgment calls the paper's equations don't pin down, made explicitly here:
- **Padding**: batched clean/corrupted inputs are padded to a common length; the GradPath
  objective `||G(path) - G(x_u')||²` is masked to non-padded positions only, so padding logits
  don't distort the descent direction.
- **Step-size floor**: `1 / ||g||₂` is clamped away from division-by-zero (`clamp_min(1e-12)`)
  for the (degenerate) case of an exactly-zero gradient at some step.

If you have the authors' own code or a more literal per-node reading in mind, flag it — the
per-input-only approximation above is the one that matches the paper's reported runtime, but it's
still an inference from the method description, not something the paper states outright.

## Sanity check

`get_scores_eap_gp`'s only genuinely new mechanic relative to EAP-IG is Step A: injecting a
`requires_grad` leaf mid-forward-pass via a hook, backpropagating a *different* objective
(`||G(·) - G(x_u')||²`, not the task metric `L`) through it, and taking a normalized step.
[`tests/test_gradpath_mechanic.py`](tests/test_gradpath_mechanic.py) exercises exactly that
pattern against a small synthetic network (no transformer_lens/GPU needed) and asserts the
objective decreases monotonically and every path point is finite and distinct:

```bash
python tests/test_gradpath_mechanic.py
```

This does **not** test `get_scores_eap_gp` itself end-to-end (that needs `transformer_lens` and,
as-written — same as every other method in this codebase — a CUDA GPU, since `attribute.py`
hardcodes `device='cuda'` for its score tensors). Run it against a real model/task before trusting
results; treat this as an implementation-from-the-paper you should verify, not a port of code the
authors published.
