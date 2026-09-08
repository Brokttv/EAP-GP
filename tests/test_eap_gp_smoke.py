"""
Integration smoke test for EAP-GP against a real model (GPT-2 small) and the
repo's own greater-than task (greater_than_data.csv, already in this repo root
since it ships with upstream EAP-IG).

Unlike tests/test_gradpath_mechanic.py (which checks the GradPath autograd
mechanic in isolation on a toy network), this runs the actual
`get_scores_eap_gp` code path end to end: real hooks on a real
HookedTransformer, real backward passes through attention/MLP layers.

Needs: transformer_lens, a CUDA GPU (this library hardcodes device='cuda' for
its score tensors in every attribution method, not just EAP-GP).

Run with:
    python tests/test_eap_gp_smoke.py

This is deliberately small (a few dozen examples) so it runs in well under a
minute and is meant to catch outright bugs (crashes, NaNs, degenerate
all-zero scores), not to reproduce the paper's reported numbers. Treat a
clean run here as "safe to move on to a real experiment," not as "the
implementation is correct" -- see the README's "Implementation notes /
assumptions" section for the judgment calls this fork made that the paper's
equations don't fully pin down.
"""
from functools import partial
import time

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), list(labels)


class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], row['label']

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def get_prob_diff(tokenizer):
    year_indices = torch.tensor([tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])

    def prob_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)[:, year_indices]
        results = []
        for prob, year in zip(probs, labels):
            results.append(prob[year + 1:].sum() - prob[:year + 1].sum())
        results = torch.stack(results)
        if loss:
            results = -results
        return results.mean() if mean else results

    return prob_diff


def run_method(model, dataloader, metric, method, **kwargs):
    g = Graph.from_model(model)
    start = time.time()
    attribute(model, g, dataloader, partial(metric, loss=True, mean=True), method=method, **kwargs)
    elapsed = time.time() - start

    scores = g.scores
    n_nan = torch.isnan(scores).sum().item()
    n_nonzero = (scores != 0).sum().item()

    g.apply_topn(50, True)
    circuit_perf = evaluate_graph(model, g, dataloader, partial(metric, loss=False, mean=False)).mean().item()

    return {
        'method': method,
        'seconds': elapsed,
        'n_nan': n_nan,
        'n_nonzero_scores': n_nonzero,
        'total_scores': scores.numel(),
        'circuit_perf_top50': circuit_perf,
    }


def main():
    model_name = 'gpt2-small'
    model = HookedTransformer.from_pretrained(model_name, device='cuda')
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ds = EAPDataset('greater_than_data.csv')
    ds.head(40)  # small subset: this is a smoke test, not a full experiment
    dataloader = ds.to_dataloader(40)
    prob_diff = get_prob_diff(model.tokenizer)

    baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
    print(f"Baseline (full model) performance: {baseline:.4f}\n")

    results = []
    results.append(run_method(model, dataloader, prob_diff, 'EAP'))
    results.append(run_method(model, dataloader, prob_diff, 'EAP-IG-inputs', ig_steps=5))
    results.append(run_method(model, dataloader, prob_diff, 'EAP-GP', ig_steps=5))

    print(f"{'method':<16} {'seconds':>8} {'NaNs':>6} {'nonzero/total':>16} {'top-50 perf':>12}")
    for r in results:
        print(f"{r['method']:<16} {r['seconds']:>8.2f} {r['n_nan']:>6} "
              f"{r['n_nonzero_scores']:>7}/{r['total_scores']:<8} {r['circuit_perf_top50']:>12.4f}")

    for r in results:
        assert r['n_nan'] == 0, f"{r['method']} produced NaN scores"
        assert r['n_nonzero_scores'] > 0, f"{r['method']} produced all-zero scores (dead computation?)"

    eap_gp = next(r for r in results if r['method'] == 'EAP-GP')
    eap_ig = next(r for r in results if r['method'] == 'EAP-IG-inputs')
    assert eap_gp['seconds'] < 20 * eap_ig['seconds'], (
        "EAP-GP took over 20x as long as EAP-IG-inputs; the paper reports ~5x -- "
        "something is likely wrong (e.g. an accidental extra loop, or retained graphs)."
    )

    print("\nOK: no NaNs, no dead (all-zero) score tensors, EAP-GP runtime is in a sane "
          "ballpark relative to EAP-IG-inputs.")
    print("This does NOT confirm the scores are *correct*, only that the code path runs "
          "and isn't obviously broken. Compare circuit_perf_top50 across methods and against "
          "your task's own faithfulness curve before trusting EAP-GP results in the paper.")


if __name__ == '__main__':
    main()
