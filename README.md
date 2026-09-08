# EAP-IG (+ EAP-GP)

> **This is a fork of [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG)** (Michael Hanna's library
> for circuit-finding in transformer LMs, MIT licensed, see `LICENSE`). Everything below the
> "Changes in this fork" section is the original upstream README, describing the base library
> this fork builds on.
>
> **What this fork adds:** an implementation of **EAP-GP** ("Edge Attribution Patching with
> GradPath"), a new attribution method from Zhang et al. 2025 that has no public code release.
> See [EAP-GP (GradPath)](#eap-gp-gradpath) below.

## Changes in this fork

Commits on top of upstream, all mine:
- [`src/eap/attribute.py`](src/eap/attribute.py): added `get_scores_eap_gp` and wired it into
  `attribute()` as `method='EAP-GP'`. The rest of this file (EAP, EAP-IG, clean-corrupted,
  information-flow-routes) is untouched upstream code.
- [`tests/test_gradpath_mechanic.py`](tests/test_gradpath_mechanic.py): a new, dependency-light
  sanity test (plain PyTorch, no `transformer_lens`/GPU needed) that isolates and checks the one
  genuinely new mechanic EAP-GP introduces.
- [`tests/test_eap_gp_smoke.py`](tests/test_eap_gp_smoke.py): a new integration smoke test that
  runs EAP, EAP-IG, and EAP-GP against real GPT-2 small on the repo's own greater-than task, and
  checks for crashes, NaNs, and dead (all-zero) score tensors. Needs `transformer_lens` and a
  CUDA GPU. See "Testing" below.
- This `README.md`: rewritten to document the fork and the new method, while keeping the
  original library's docs intact below.

## EAP-GP (GradPath)

From:

> Zhang, Dong, Zhang, Yang, Hu, Liu, Zhou, Wang. *EAP-GP: Mitigating Saturation Effect in
> Gradient-based Automated Circuit Identification*. [arXiv:2502.06852](https://arxiv.org/abs/2502.06852) (2025).

Both EAP-IG and EAP-GP estimate every edge's attribution score with the same first-order formula:

$$
\text{score}(u, v) = (x_u - x_u') \cdot \frac{1}{k}\sum_{j=1}^{k} \frac{\partial L(\text{path}(j))}{\partial x_v}
$$

where $x_u$ / $x_u'$ are node $u$'s clean/corrupted activations, and the mean is taken over $k$
points along a path between them. This fork, like upstream's cheap `EAP-IG-inputs` variant,
applies the path only at the model's input activations and lets it propagate through the rest of
the network. That's what makes the method $O(k)$ forward/backward passes instead of $O(k \cdot n_{layers})$.

The two methods differ only in how `path(j)` is chosen:

**EAP-IG** uses a straight line between the corrupted and clean activations:

$$
\text{path}(j) = x_u' + \frac{j}{k}(x_u - x_u')
$$

**EAP-GP** replaces that line with a *GradPath*. It starts at the clean activation, then takes $k$
unit-normalized gradient-descent steps that pull the path toward matching the *corrupted* run's
full output $G(x_u')$:

$$
\text{path}(0) = x_u
$$

$$
g_{j+1} = \nabla_{\text{path}(j)} \left\lVert G(\text{path}(j)) - G(x_u') \right\rVert_2^2
$$

$$
\text{path}(j+1) = \text{path}(j) - \frac{g_{j+1}}{\left\lVert g_{j+1} \right\rVert_2}
$$

The intuition (Section 5 of the paper): a fixed straight-line path can land in regions where
$\partial L/\partial x_v \approx 0$ (saturation), silently zeroing out an edge's estimated
importance even when the edge matters. Since GradPath actively moves toward the corrupted output,
it's constructed to avoid exactly those flat regions.

This changes the *points* the gradient is averaged over. The outer $(x_u - x_u')$ term and the
edge-score bookkeeping (via `make_hooks_and_matrices`'s forward/backward hooks) are untouched
from EAP-IG.

### Usage

```python
from eap.attribute import attribute

scores = attribute(
    model, graph, dataloader, metric,
    method='EAP-GP',
    ig_steps=5,   # k in the paper; they find k=5 works best (ablated over k in [3, 20])
)
```

Otherwise follows the same `Graph` / dataloader / metric conventions as the rest of this
library, see "How to use this library" below and `ioi.ipynb` / `greater_than.ipynb` for the
surrounding workflow.

### Implementation notes / assumptions

The paper's equations describe the update per node: for any upstream activation $x_u$, they define
$G(s) = G(E(z) - x_u + s)$, the model's output with just that one node patched. Taken literally,
that would mean running a separate $k$-step gradient descent for *every* node in the graph.

**Why that can't be the real implementation.** A separate GradPath per node would be intractable,
and it doesn't match the paper's own numbers: they report EAP-GP at roughly 5x the wall-clock cost
of EAP-IG on GPT-2 small / IOI, not the many-orders-of-magnitude blowup a true per-node path would
cause. Upstream EAP-IG faces the identical tension and solves it the same way in its cheap variant:
build one joint path at the model's *input* activations, and reuse the standard first-order-Taylor
edge bookkeeping to get every downstream edge's score from that single path in one pass.

**What this implementation does.** It applies that same trick to GradPath:
1. Build one GradPath over the input activations (Step A: `steps` extra forward + backward passes).
2. Reuse that path for every edge's attribution (Step B: `steps` more forward + backward passes).

Two passes per step (instead of EAP-IG's one) is consistent with the paper's reported ~5x overhead.

**Padding.** Batched clean/corrupted inputs are padded to a common length. The GradPath objective
$\lVert G(\text{path}) - G(x_u') \rVert^2$ is masked to non-padded positions only, so padding
logits don't distort the descent direction.

**Step-size floor.** The update divides by $\lVert g \rVert_2$; this is clamped away from zero
(`clamp_min(1e-12)`) to avoid a division-by-zero in the degenerate case of an exactly-zero
gradient at some step.

If you have the authors' own code, or a more literal per-node reading in mind, that would
supersede the reading above. The per-input-only approximation here is an inference from the
method description, matched to the paper's reported runtime, not something the paper states
outright.

### Testing

There are two tests, covering two different things. Run both before trusting EAP-GP results in
an experiment, and definitely before handing this off to a collaborator to run at scale.

**1. The GradPath mechanic, in isolation.** `get_scores_eap_gp`'s only genuinely new piece
relative to EAP-IG is Step A: injecting a `requires_grad` leaf mid-forward-pass via a hook,
backpropagating a *different* objective ($\lVert G(\cdot) - G(x_u') \rVert^2$, not the task
metric $L$) through it, and taking a normalized step.
[`tests/test_gradpath_mechanic.py`](tests/test_gradpath_mechanic.py) exercises exactly that
pattern against a small synthetic network (plain PyTorch, no transformer_lens or GPU needed) and
asserts the objective decreases monotonically and every path point is finite and distinct:

```bash
python tests/test_gradpath_mechanic.py
```

This only checks the autograd trick works in principle. It does **not** touch a real model, so it
can't catch bugs in how the hooks interact with a real `HookedTransformer`'s actual attention/MLP
graph.

**2. `get_scores_eap_gp` against a real model.** [`tests/test_eap_gp_smoke.py`](tests/test_eap_gp_smoke.py)
runs EAP, EAP-IG-inputs, and EAP-GP on GPT-2 small against the repo's own greater-than task
(`greater_than_data.csv`, already in this repo). Needs `transformer_lens` and a CUDA GPU (every
method in this library hardcodes `device='cuda'` for its score tensors, not just EAP-GP):

```bash
python tests/test_eap_gp_smoke.py
```

It checks the basics: no crash, no NaNs, no all-zero (dead) score tensor, and that EAP-GP's
runtime is in a sane ballpark relative to EAP-IG (the paper reports ~5x; the test only flags
something above 20x, as a loose tripwire rather than a tight bound). If any of those fail, there's
a real bug, don't proceed to a full experiment.

If it passes, that still only means the code *runs*, not that the scores are *correct*. Before
trusting numbers for the paper:
- Compare the printed `circuit_perf_top50` for EAP-GP against EAP-IG-inputs at the same top-n. On
  the paper's own GPT-2-small results, EAP-GP should do as well as or better than EAP-IG at
  matched circuit sizes; a much worse result is a signal something in the implementation (likely
  the per-input-only approximation described above) doesn't hold on your task.
  It's also fine, and worth doing, to run the same check on the residual-stream-node-level task
  you actually care about for the paper, not just this smoke test's greater-than task.
- Try a couple of different `ig_steps` values (the paper explores $k \in [3, 20]$, and reports
  $k=5$ as their main setting). Wildly unstable results across nearby $k$ would be a red flag.
- If you want a closer check against the paper's own headline numbers (GPT-2 small, IOI, ~80%
  NFS for EAP-GP vs. ~62% for EAP-IG at 97.5% sparsity), you'll need to set up the IOI task and
  the Normalized Faithfulness Score metric yourself. This repo ships a greater-than dataset out of
  the box, not IOI-for-GPT-2, so that comparison needs more setup than the smoke test above.

