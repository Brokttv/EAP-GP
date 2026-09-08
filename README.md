# EAP-IG (+ EAP-GP)

> **This is a fork of [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG)** (Michael Hanna's library
> for circuit-finding in transformer LMs, MIT licensed — see `LICENSE`). Everything below the
> "Changes in this fork" section is the original upstream README, describing the base library
> this fork builds on.
>
> **What this fork adds:** an implementation of **EAP-GP** ("Edge Attribution Patching with
> GradPath"), a new attribution method from Zhang et al. 2025 that has no public code release.
> See [EAP-GP details](#eap-gp-gradpath) below.

## Changes in this fork

Commits on top of upstream, all mine:
- [`src/eap/attribute.py`](src/eap/attribute.py) — added `get_scores_eap_gp` and wired it into
  `attribute()` as `method='EAP-GP'`. The rest of this file (EAP, EAP-IG, clean-corrupted,
  information-flow-routes) is untouched upstream code.
- [`tests/test_gradpath_mechanic.py`](tests/test_gradpath_mechanic.py) — a new, dependency-light
  sanity test (plain PyTorch, no `transformer_lens`/GPU needed) that isolates and checks the one
  genuinely new mechanic EAP-GP introduces.
- This `README.md` — rewritten to document the fork and the new method, while keeping the
  original library's docs intact below.

## EAP-GP (GradPath)

From:

> Zhang, Dong, Zhang, Yang, Hu, Liu, Zhou, Wang. *EAP-GP: Mitigating Saturation Effect in
> Gradient-based Automated Circuit Identification*. [arXiv:2502.06852](https://arxiv.org/abs/2502.06852) (2025).

Both EAP-IG and EAP-GP estimate every edge's attribution score with the same first-order formula

```
score(u, v) = (x_u - x_u') · mean_j[ ∂L(path(j)) / ∂x_v ]
```

where `x_u`/`x_u'` are node `u`'s clean/corrupted activations and the mean is taken over `k`
points along a path between them (this fork, like upstream's cheap `EAP-IG-inputs` variant,
applies the path only at the model's input activations and lets it propagate through the rest of
the network — that's what makes the method O(k) forward/backward passes instead of O(k · n_layers)).

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
library — see "How to use this library" below and `ioi.ipynb` / `greater_than.ipynb` for the
surrounding workflow.

### Implementation notes / assumptions

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

If you have the authors' own code or a more literal per-node reading in mind, that would supersede
the reading above — the per-input-only approximation is an inference from the method description
(matched to the paper's reported runtime), not something the paper states outright.

### Sanity check

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
as-written — same as every other method in this library — a CUDA GPU, since `attribute.py`
hardcodes `device='cuda'` for its score tensors). Run it against a real model/task before trusting
results; treat this as an implementation-from-the-paper you should verify, not a port of code the
authors published.

---

# Upstream: EAP-IG

This library contains various resources for finding circuits in autoregressive transformer LMs. At a high level, a circuit is the part of your model responsible for performing a given task; all nodes / edges outside the circuit can be corrupted without harming model performance. For more on circuits, see [this paper](https://arxiv.org/abs/2403.17806 ) or [this paper](https://arxiv.org/abs/2403.19647). For a demo of this library's features, check out `greater_than.ipynb`; for a demo using larger models (Llama-3 8B), check out `ioi.ipynb`.

This library has tools that will let you do a variety of things:
- Construct a `Graph` object representing the computational graph of most autoregressive transformer LMs in the [TransformerLens library](https://github.com/TransformerLensOrg/TransformerLens). Computational graphs can be drawn at the following levels:
    - **Node and edge (default)**: Nodes are model components (attention heads and MLPs), and edges are connections between them (across layers, via the residual stream)
    - **Node**: The graph contains only nodes, and we disregard the edges. This is equivalent to saying that for every node in the circuit, all of its outgoing edges are also in the circuit.
    - **Neuron**: The graph contains only nodes, split into neurons. That is, you can include individual neurons, or output dimensions of a given component.
- Use attribution-based circuit-finding methods to produce scores (indirect effect estimates) for each node or edge in the computational graph. The attribution methods supported are:
    - [Edge Attribution Patching (EAP)](https://arxiv.org/abs/1703.01365): Computes a first-order approximation of the indirect effect of each edge, i.e. the amount that your loss changes upon corrupting the edge. Essentially multiplies the change in component outputs by the gradient on clean inputs. Runs in O(1) time. See the [original blog post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) for more info.
    - [Edge Attribution Patching with Integrated Gradients (EAP-IG, inputs)](https://arxiv.org/abs/2403.17806): An adaptation of EAP that improves circuit quality by averaging the gradient computation over *m* steps taken between the clean and corrupted inputs, as in the [integrated gradients paper](https://arxiv.org/abs/1703.01365). Takes O(*m*) time.
    - [Edge Attribution Patching with Integrated Gradients (EAP-IG, activations)](): Another adaptation of EAP using integrated gradients; instead of taking the gradient when the input embeddings are interpolated between the clean and corrupted inputs, it interpolates between the clean/corrupted activations for each component. This takes longer (O(*m * L*) time, given an *L*-layer model), but is somewhat more principled, and allows for estimating zero / mean ablation effects as well (just like EAP).
    - **Edge Attribution Patching with GradPath (EAP-GP)**: added in this fork — see above.
    - [Clean-Corrupted](https://arxiv.org/abs/2403.17806): A variant of EAP/-IG that takes the gradient at two steps: the clean and corrupted input.
- Use either a greedy-search or top-n approach to find a circuit of a given size based on these scores
- Evaluate your circuit's performance (allowing you to compute its faithfulness)

## How to install this library
To use this library, just install it using `pip install .`. If you'd like to be able to visualize the graphs you create, please use the `viz` option: `pip install .[viz]`. This may require you to install graphviz:

<details>
<summary>How to install graphviz</summary>

**MacOS**
```bash
brew install graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
pip install . # or `uv sync`
```

**Ubuntu**
```bash
apt-get update
apt-get install -y graphviz libgraphviz-dev build-essential
```

For other operating systems or if you encounter build errors, ensure the Graphviz C libraries are installed and accessible to the build system via environment variables (like `CFLAGS` and `LDFLAGS`).
</details>

## How to use this library
For a demo of this library's features, check out `greater_than.ipynb`; for a demo using larger models (Llama-3 8B), check out `ioi.ipynb`. In general, the circuit-finding pipeline looks like this:
- Define a task with clean and corrupted inputs, a label associated with the clean inputs, and a metric measuring model performance. (`dataloader = EAPDataset('greater-than').to_dataloader()`, `metric = ...`)
- Define your model's computation graph at the desired level of granularity. (`graph = Graph.from_model(model)`)
- Use an attribution method to estimate the change in the metric that would occur if you were to corrupted / mean-ablate / zero-ablate each unit in your computation graph (i.e., estimate each unit's indirect effect). (`attribute(model, graph, dataloader, metric, method='EAP-IG-inputs', ig_steps=5)`)
- Using the indirect effects / scores calculated, define a circuit by taking the top-n edges / nodes / neurons of your graph. (`graph.apply_topn(n)`)
- Evaluate your circuit's performance, recording the metric when you actually corrupt / ablate all edges / nodes / neurons not in the circuit. (`results = evaluate_graph(model, graph, dataloader, metric)`)

## FAQs
- **How is the computation graph drawn?**: In this library, graphs are defined as being collections of nodes and edges, where nodes are either the inputs, attention heads, MLPs, or logits. Edges connect nodes across layers, accounting for the fact that nodes can engage in cross-layer communication via the residual stream. Each MLP (and the logits) has 1 input, but each attention head has 3: the Q, K, and V input.
- **Which models are compatible with this library?**: In general, this library works with autoregressive transformer LMs in TransformerLens. It's important that models use pre-LayerNorm, as post-LayerNorm means that the residual stream is no longer a sum of all previous components. The models I have used so far are: GPT-2, Pythia, Mistral, Qwen, OLMo, Llama, and Gemma (using a workaround / hack since there is a post layer-norm that doesn't totally destroy the residual stream.)
- **What about models with Grouped Query Attention (GQA)?**: To work with these models, please ungroup the GQA by setting `model.cfg.ungroup_grouped_query_attention = True`; this will remove all of the efficiency benefits of GQA, but allow the model to be used with this library.
- **What about zero and mean ablations?**: I think these are often best avoided, at least zero-ablation. But these are supported as well (with EAP / EAP-IG (activations)). Just set the `intervention` argument of `attribute` and `evaluate_graph` to `zero`, `mean`, or `mean-positional`; in the latter case, all inputs must have the same length / structure. You can specify the dataloader to take the mean over via the `intervention_dataloader` argument.

## More Info
This library contains the following files:
- `graph.py` contains the Node, Edge, and Graph classes.
- `attribute.py` contains the implementation of EAP/-IG (and, in this fork, EAP-GP)
- `attribute_node.py` contains the implementation of EAP/-IG, but for nodes / neurons
- `evaluate.py` contains code for evaluating circuits
- `visualization.py` contains code for choosing colors / controlling how circuits are visualized

This repo owes a lot to:
- [The original ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery), in particular for its conceptualization of the graph and its visualization—go check it out!
- [Aaquib Syed's original EAP implementation](https://github.com/Aaquib111/edge-attribution-patching/tree/minimal-implementation), for its memory efficient implementation of EAP
