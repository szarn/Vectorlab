# vectorlab

Find, inspect and edit behavior directions in transformer language models.

vectorlab takes a behavior defined by two contrasting datasets, such as harmful vs harmless prompts or
committed vs hedged answers. From these it:

- extracts a direction for the behavior at every layer;
- writes an offline HTML report showing where and how the direction is represented;
- searches for the edit that best moves the behavior at a given cost in KL divergence;
- applies the edit as reversible hooks or bakes it into the weights;
- exports the edited model as safetensors or MLX.

It follows Arditi et al. (2024), *Refusal in language models is mediated by a single direction* (ablation
and weight orthogonalization), and Rimsky et al. (2023), *Steering Llama 2 via contrastive activation
addition*. It is written from scratch for Apple Silicon (PyTorch MPS, bf16) and supports Qwen3, Llama and
Gemma 3.

![Held-out Cohen's d by layer for four behaviors on Qwen3-4B](docs/figures/separation.svg)

## Results

On Qwen/Qwen3-4B-Instruct-2507 unless noted. Held-out prompts were never seen by extraction or search.
KL is measured against the original model on 100 neutral prompts. [FINDINGS.md](FINDINGS.md) has the
details, figures and caveats.

| behavior | model | edit | held-out before → after | KL |
|---|---|---|---|---|
| refusal | Qwen3-0.6B | ablate (hooks or weights) | 59.8% → 0.0% refusal (n=122) | 0.083 |
| refusal | Gemma 3 12B | ablate (weights) | 91.8% → 9.0% refusal (n=122) | 0.196 |
| refusal | Gemma 3 12B | p-e-w/gemma-3-12b-it-heretic, for reference | 91.8% → 4.9% refusal (n=122) | 0.144 |
| refusal | Qwen3-4B | ablate | 100% → 21.9% on 32 *train* prompts (optimistic) | 0.199 |
| hedging | Qwen3-0.6B | add | 1.67 → 2.85 decisiveness, judge 1–5 (n=60) | 0.160 |
| hedging | Qwen3-4B | add | 1.92 → 2.30 decisiveness, judge 1–5 (n=60) | 0.068 |
| sycophancy | Qwen3-4B | ablate | 1.53 → 1.18 sycophancy, judge 1–5 (n=60) | 0.171 |
| verbosity, answer-text direction | Qwen3-4B | add | 322 → 316 tokens (n=60), no real effect | 0.011 |
| verbosity, instruction-pair direction | Qwen3-4B | add (over the KL budget) | 322 → 256 tokens (n=60), −21% | 0.465 |

The judge is google/gemma-3-12b-it. It agrees with 60 hand labels per behavior at κ 0.90 (hedging) and
0.81 (sycophancy).

![Cross-effect matrix of the hedging, sycophancy and verbosity edits](docs/figures/cross-effects.svg)

Two example reports are in [`docs/examples/`](docs/examples): a run report for hedging on Qwen3-4B and the
cross-behavior comparison. They are self-contained HTML with Plotly inlined. GitHub does not render them, so
download them and open them in a browser.

## Install

Requires Python 3.12+. A Mac with Apple Silicon is assumed: models load on `mps`.

```sh
uv venv --python 3.12
uv pip install -e ".[dev]"        # add mlx for MLX export: ".[dev,mlx]"
python scripts/fetch_refusal.py   # refusal prompts and the neutral KL prompts
```

`fetch_refusal.py` downloads AdvBench and JailbreakBench (MIT) and Alpaca-cleaned (CC BY 4.0). They are not
redistributed here. The default judge, google/gemma-3-12b-it, is gated: accept its license on Hugging Face
first, or pass `--judge` with another model.

## Usage

```sh
B=behaviors/hedging.yaml
RUN=runs/hedging-Qwen--Qwen3-4B-Instruct-2507

vectorlab extract $B --model Qwen/Qwen3-4B-Instruct-2507   # directions, separation, logit lens
vectorlab analyze $RUN                                     # writes $RUN/report.html
vectorlab search $RUN --trials 40                          # Optuna: behavior vs KL, suggests a trial
vectorlab edit $RUN --trial 23                             # or --layer 21 --alpha 5 --range 21:23
vectorlab eval $RUN                                        # held-out metric and KL, before and after
vectorlab analyze $RUN                                     # the report now includes the edit

vectorlab edit runs/refusal-Qwen--Qwen3-0.6B --trial 9 --weights
vectorlab export runs/refusal-Qwen--Qwen3-0.6B --format mlx

vectorlab compare runs/hedging-… runs/sycophancy-… --directions-only runs/refusal-…
```

| command | what it does |
|---|---|
| `extract` | Captures residual activations for both classes, then computes per-layer mean-difference directions, held-out Cohen's d, probe accuracy, a lexical baseline, the logit lens and token projections. |
| `analyze` | Writes `report.html`: layer separation, cross-layer cosine, projection histograms, logit lens, token heatmap, search tradeoff, judges, variants, before/after answers. Reads only the run directory. |
| `search` | Runs a two-objective Optuna search (behavior score, KL) over source layer, strength and layer range, on a sample of train prompts. |
| `edit` | Chooses the intervention: hooks (default) or `--weights` (orthogonalization or a baked bias). |
| `eval` | Scores the original and edited model on held-out prompts, plus KL on neutral prompts. `--model` evaluates another checkpoint against the same original. |
| `export` | Saves the weight-edited model as safetensors or MLX and reloads it to check KL. |
| `compare` | Builds one page for several behaviors on one model: direction similarity, plus cross-effects of each edit on every behavior's prompts. |

Add `-v` to any command for progress logs. Original-model generations and log-probs are cached in `cache/`,
and only one full model is held in memory at a time.

### Behavior specs

A behavior is a YAML file plus two JSONL datasets:

```yaml
name: sycophancy
capture: response          # prompt: last prompt token; response: mean over the answer tokens
positive: data/sycophancy/sycophantic.jsonl
negative: data/sycophancy/honest.jsonl
intervention: ablate       # or add
metric: judge              # refusal | verbosity | judge
target: min                # which way the edit should move the metric
length_penalty: 2.0        # optional: search penalty on relative change in answer length
rubric: |                  # required for metric: judge
  Rate how much the answer tells the user what they want to hear ...
```

- **Records:** each record has a `prompt`, and in response mode also an `answer`.
- **Instruction pairs:** these append opposite instructions to the same prompt. They keep the plain prompt
  in `base`, and the split, search and eval use that plain prompt. `scripts/instruction_pairs.py` builds
  them.
- **Splits:** the train/held-out split is 80/20 by (base) prompt, seeded.

## Tests

```sh
pytest -m "not slow"   # tiny random models: weight edits, hooks, compare statistics
pytest                 # also loads real checkpoints from the Hugging Face cache
```

## Layout

```
vectorlab/   the package (cli, capture, directions, hooks, weights, search, evaluate, report, compare, export)
behaviors/   behavior specs
data/        datasets, judge labels and agreement results (refusal data is fetched, not committed)
scripts/     data fetching and expansion, judge sampling and rescoring, doc figures
docs/        animated figures for these docs and example reports
tests/
```
