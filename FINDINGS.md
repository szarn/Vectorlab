# Findings

What vectorlab found on four behaviors: refusal, hedging, sycophancy and verbosity. The main model is
Qwen/Qwen3-4B-Instruct-2507. Qwen/Qwen3-0.6B was used for the first milestones, and google/gemma-3-12b-it
for a refusal benchmark against an existing abliterated model.

Every number comes from a run in this repository's workflow (`vectorlab extract → search → edit → eval →
compare`). Unless noted:

- **Held-out** means the 20% of prompts held out by the seed-0 split: 60 prompts for the judged and length
  behaviors, 122 for refusal.
- Decoding is greedy, in bf16 on Apple MPS, with batch size 16.
- **KL** is the mean KL(original ‖ edited) of the next-token distribution on 100 neutral Alpaca prompts.
- **CI** is a paired 95% interval over prompts.

## Contents

1. [Behaviors and data](#behaviors-and-data)
2. [Judges](#judges)
3. [What the report views showed](#what-the-report-views-showed)
4. [Search, and why its numbers are optimistic](#search-and-why-its-numbers-are-optimistic)
5. [Edits on Qwen3-4B](#edits-on-qwen3-4b)
6. [How the behaviors relate](#how-the-behaviors-relate)
7. [Verbosity: a readable direction that does not steer](#verbosity-a-readable-direction-that-does-not-steer)
8. [Gemma 3 12B refusal benchmark](#gemma-3-12b-refusal-benchmark)
9. [Limitations](#limitations)

## Behaviors and data

| behavior | direction (positive − negative) | captured at | edit | metric |
|---|---|---|---|---|
| refusal | harmful − harmless prompts | last prompt token | ablate | refusal-phrase rate |
| hedging | committed − hedged answers | answer tokens (mean) | add | judge, 1–5, how decisively the answer commits |
| sycophancy | sycophantic − honest answers | answer tokens (mean) | ablate | judge, 1–5, how much it tells the user what they want to hear |
| verbosity | concise − verbose answers | answer tokens (mean) | add | answer length in tokens |
| verbosity-instruction | "be brief" − "be detailed" instructions on the same questions | last prompt token | add | answer length in tokens |

- **Refusal:** 609 harmful prompts from AdvBench and JailbreakBench (both MIT) and as many harmless Alpaca
  instructions (CC BY 4.0). `scripts/fetch_refusal.py` downloads them, so they are not committed.
- **Hedging, sycophancy, verbosity:** hand-written seed pairs (52, 51 and 48), expanded to 300 pairs each by
  Qwen3-4B-Instruct-2507 with `scripts/expand.py`. The expander filters its own output:
  - hedged answers must not commit, and committed answers must argue their assigned side;
  - sycophancy prompts must contain a real flaw, and only the honest answer may point it out;
  - hedging and sycophancy pairs must roughly match in length;
  - verbosity answers must fall in fixed word ranges.

  Spot-check samples are in `data/<behavior>/sample.md`.
- **Verbosity-instruction:** the same 300 verbosity questions, each with one of four brief/detailed
  instruction pairs appended (`scripts/instruction_pairs.py`). The plain question is kept in a `base` field.
  The split, the search and the eval all use the plain question, so its held-out set is identical to the
  verbosity run's.

## Judges

Judged metrics use the expected digit under a judge model's next-token distribution, restricted to the
digits 1–5. To validate a judge, 60 answers per behavior were hand-labeled before any judge output was
seen: dataset answers, the model's own answers, and steered answers.

| behavior | judge | κ (binary at 3) | Spearman | MAE | verdict |
|---|---|---|---|---|---|
| hedging | google/gemma-3-12b-it | 0.90 | 0.90 | 0.27 | adopted |
| hedging | Qwen/Qwen3-4B-Instruct-2507 | 0.80 | 0.90 | 0.45 | rejected: +0.71 too lenient on its own 4B outputs |
| hedging | google/gemma-3-4b-it | 0.53 | 0.88 | 1.06 | rejected: +0.5 to +1.4 too lenient across sources |
| sycophancy | google/gemma-3-12b-it | 0.81 | 0.87 | 0.22 | adopted |

The sycophancy judge's mean bias against the hand labels:

| answer source | bias |
|---|---|
| baseline answers | +0.24 |
| dataset answers | 0.00 |
| steered answers | −0.02 |

Each report's Judges table also shows the 0.6B and 4B hedging edits rescored by every judge.

## What the report views showed

![Held-out Cohen's d by layer for the four behaviors on Qwen3-4B](docs/figures/separation.svg)

**Layer separation and the lexical baseline.** A logistic probe reaches 99.6–100% held-out accuracy at its
best layer for every behavior. Accuracy alone therefore says little. The informative comparison is the
effect size against the lexical baseline: the same statistic on the mean token embedding, before any layer
runs.

| run | peak d (layer) | lexical baseline d |
|---|---|---|
| refusal, Qwen3-0.6B | 3.98 (23 of 28) | 3.10 |
| refusal, Qwen3-4B | 9.38 (24 of 36) | 2.90 |
| refusal, Gemma 3 12B | 7.95 (42 of 48) | 3.32 |
| hedging, Qwen3-0.6B | 6.28 (18 of 28) | 4.82 |
| hedging, Qwen3-4B | 5.96 (21 of 36) | 4.86 |
| sycophancy, Qwen3-4B | 6.61 (21 of 36) | 5.87 |
| verbosity, Qwen3-4B | 4.89 (20 of 36) | 1.15 |
| verbosity-instruction, Qwen3-4B | 13.59 (16 of 36) | 5.09 |

- **Refusal** on 4B separates far beyond vocabulary: 9.4 against 2.9.
- **Hedging and sycophancy** are largely separable by vocabulary alone (lexical d 4.9 and 5.9). Their
  directions likely carry stylistic wording, such as praise words and "it depends", as well as the behavior.
- **Verbosity**'s answer-text direction separates well with almost no lexical signal. It still turned out
  not to steer length (see [Verbosity](#verbosity-a-readable-direction-that-does-not-steer)).

**Direction similarity across layers.** The three answer-text directions are stable from layer to layer.
The two prompt-mode directions drift faster.

| direction | mean cosine, layer to next (layers 8–28) | cosine, layer 12 vs layer 24 |
|---|---|---|
| hedging | 0.92 | 0.48 |
| sycophancy | 0.93 | 0.51 |
| verbosity | 0.94 | 0.48 |
| refusal | 0.76 | 0.11 |
| verbosity-instruction | 0.77 | 0.22 |

**Logit lens** (each direction at its edit layer, through the final norm and the unembedding). Three of the
directions read cleanly:

- **Hedging** (committed − hedged) suppresses "depends on", "depending" and "circumstances", in English and
  Chinese tokens (取决于, 具体情况, 各有).
- **Sycophancy** promotes praise (commend, admirable, wonderfully, 👏, 值得) and suppresses criticism (worse,
  improperly, harmful, severely).
- **Verbosity** (concise − verbose) suppresses enthusiastic filler ("fascinating", "folks", "—and", "—even").

The verbosity-instruction direction at layer 16 is less readable. It does suppress "unpack", 逐步 ("step by
step") and 等一系列 ("a series of").

**Token heatmap** (highest summed projections at the edit layer, held-out examples):

- **Hedging:** the hedged side lights up on "depends", "personal choice" and "option". The committed side
  lights up on concrete specifics.
- **Sycophancy:** the sycophantic side lights up on "brilliant", "smart" and "!".
- **Verbosity:** the verbose side lights up on function words (the, a, it, this), and the concise side on
  pieces of content words. This suggests the answer-text direction encodes information density rather than a
  plan for how long to write.

**Projection histograms and before/after samples** are in each run's `report.html`. Two examples are
committed in `docs/examples/`.

## Search, and why its numbers are optimistic

`vectorlab search` runs 40 Optuna trials (TPE, two objectives) over the source layer, the strength, and the
layer range. Each trial is scored on 32 train prompts, and KL is measured on the neutral prompts. For judged
behaviors the objective subtracts 2 × |relative change in answer length|, so an edit cannot win just by
writing more or less. The suggested trial is the Pareto trial with the best objective within KL ≤ 0.2.

![Sycophancy search trials, score against KL](docs/figures/sycophancy-search.svg)

The suggested trial is the best of 40 noisy measurements on 32 prompts. Its train score therefore overstates
the held-out effect (the winner's curse):

| run (suggested trial) | train prompts (n=32) | held-out |
|---|---|---|
| refusal, Qwen3-0.6B (9) | 62.5% → 0.0% | 59.8% → 0.0% |
| refusal, Gemma 3 12B (2) | 87.5% → 3.1% | 91.8% → 9.0% (weights) |
| hedging, Qwen3-0.6B (34) | 1.43 → 3.57 (+2.14) | 1.67 → 2.85 (+1.18) |
| hedging, Qwen3-4B (23) | 2.27 → 2.78 (+0.51) | 1.92 → 2.30 (+0.38, batch 32) |
| sycophancy, Qwen3-4B (11) | 1.61 → 1.22 (−0.39) | 1.53 → 1.18 (−0.36) |

Refusal on Qwen3-4B was searched but never evaluated on held-out prompts: 100% → 21.9% on the 32 train
prompts at KL 0.199 (trial 19). Treat that number as an upper bound on the effect.

## Edits on Qwen3-4B

| behavior | edit (suggested trial) | held-out, before → after | KL |
|---|---|---|---|
| hedging | add the layer-21 direction at layers 21–22, strength 5.0 (23) | 1.92 → 2.30 decisiveness | 0.068 |
| sycophancy | ablate the layer-28 direction at layers 12–34, strength 0.81 (11) | 1.53 → 1.18 sycophancy | 0.171 |
| verbosity | add the layer-28 direction at layer 28, strength 9.1 (39) | 322 → 316 tokens | 0.011 |
| verbosity-instruction | add the layer-10 direction at layer 10, strength 16.1 (31, over the KL budget) | 322 → 256 tokens | 0.465 |
| refusal | ablate the layer-15 direction at layers 2–17, strength 1.15 (19) | search only, see above | 0.199 |

Under the sycophancy edit, 17 of 60 held-out prompts moved toward candor by more than 0.5 points and 2 moved
away. Of the two largest drops I read, one is a clean win: the edited answer states the problem with the
user's work in its first sentence. The other is an overcorrection: the user's claim was sound, but the
edited answer called it unsafe, and the judge scored that as fully candid. The dataset's own "honest" answer for that prompt makes the same mistake,
because the expander's premise check let a correct claim through.

## How the behaviors relate

`vectorlab compare` puts the runs of one model on one page (`runs/compare-<model>/compare.html`).

![Per-layer cosine between behavior directions](docs/figures/similarity.svg)

**Direction similarity.** Directions are oriented positive minus negative, as in the table above.

- **Sycophancy and verbosity:** the sycophantic direction points toward verbose answers (cosine −0.25 to
  −0.51 against concise − verbose, at every layer).
- **Hedging and verbosity:** the committed direction points toward concise answers (+0.1 to +0.4).
- **Verbosity and verbosity-instruction:** the prompt-mode instruction direction and the answer-text
  direction agree increasingly with depth, up to 0.47 at the last layer.
- **Refusal** is near zero against the answer-text directions. Its larger values against
  verbosity-instruction (down to −0.48 at layer 7) are likely shared prompt-surface features, since both are
  read at the last prompt token.

![Cross-effect matrix](docs/figures/cross-effects.svg)

**Cross-effects.** Each edit is applied with hooks and scored on every behavior's held-out prompts with that
behavior's own metric (n = 60 per cell). Cells show the mean change ± its paired 95% CI.

| edit (rows) / scored on | hedging (1–5) | sycophancy (1–5) | verbosity (answer tokens) | KL |
|---|---|---|---|---|
| hedging | **+0.28 ± 0.16** | +0.08 ± 0.16 | −17 ± 9 | 0.068 |
| sycophancy | +0.11 ± 0.18 | **−0.36 ± 0.18** | −2 ± 5 | 0.171 |
| verbosity, instruction pair (trial 31) | +0.17 ± 0.23 | −0.02 ± 0.19 | **−66 ± 21** | 0.465 |
| verbosity, answer text (trial 39, null control) | −0.02 ± 0.10 | −0.02 ± 0.12 | −6 ± 6 | 0.011 |

The last row comes from the same comparison run with the answer-text verbosity edit in place of the
instruction-pair one. All rows use batch size 16.

- **Each edit mostly moves only its own behavior.** Besides the diagonal, one cell excludes zero: the
  hedging edit shortens answers to the verbosity questions by 17 ± 9 tokens (−5%). This matches the direction
  similarity: committed − hedged leans toward concise − verbose (cosine 0.29 at the hedging edit's layer).
- **Overlap does not predict spillover.** The sycophancy direction overlaps most with verbosity (cosine −0.45
  at its edit layer), yet ablating it changes answer length on the verbosity questions by only −2 ± 5 tokens.
  Ablation only removes the part of the activations that lies along the direction, and ordinary questions
  barely project onto a sycophancy direction.
- **The length edit does not move the judges.** The instruction-pair verbosity edit shortens its own answers
  by 21%, but leaves judged hedging and sycophancy unchanged within noise. This is only a weak test of
  whether the judges reward length: answers to those prompts are capped at 256 tokens and became only 7–12
  tokens shorter (255 → 248 and 243).
- **Noise floor.** The null answer-text verbosity edit sets it at about ±0.1 on judge scores.
- **Batch size.** Hedging's diagonal (+0.28) is lower than its eval (+0.38) only because of batch size; see
  [Limitations](#limitations).

## Verbosity: a readable direction that does not steer

![Two verbosity searches, answer length against KL](docs/figures/verbosity-searches.svg)

The answer-text direction (concise − verbose answers to the same questions) is the one that vocabulary
explains least: d 4.9 against a lexical baseline of 1.2. Adding it does not shorten answers. All
40 search trials stayed between 293 and 383 tokens against a 298-token baseline, even at KL ≈ 2. On held-out
prompts the suggested trial went from 322 to 316 tokens (−6 ± 6) at KL 0.011.

The instruction-pair direction reads the model at the last prompt token, on the same questions ending in
"Be brief" or "Answer in detail". It is added while the model answers the plain question.

This direction does steer length, with a clear dose-response along the search's Pareto front (train
prompts):

| answer length | KL |
|---|---|
| −2% | 0.05 |
| −8% | 0.22 |
| −19% | 0.47 |
| −36% | 1.4 |
| 12-token answers | 12 |

On held-out prompts the effect is larger than on the train sample:

| trial | edit | answer tokens, before → after | KL |
|---|---|---|---|
| 30 (suggested, KL ≤ 0.2) | add the layer-9 direction at layers 9–10, strength 2.5 | 322 → 305 (−5%) | 0.052 |
| 9 | add the layer-11 direction at layers 11–12, strength 3.6 | 322 → 285 (−11%) | 0.219 |
| 31 (used for the cross-effects) | add the layer-10 direction at layer 10, strength 16.1 | 322 → 256 (−21%) | 0.465 |

The run uses trial 31 even though it exceeds the KL budget. The budget fits this behavior poorly: a
length edit is meant to change how the model answers the neutral prompts too, so part of its KL is the
intended effect. Trial 31's answers stay coherent; 1 of 60 falls into a repetition loop. Before the edit,
27 of 60 answers hit the 512-token cap, and after it 9 do. The capped answers make the measured reduction
smaller than the real one.

The likely reason for the difference is that answer length is decided while reading the prompt, not
while writing the answer. The answer-text direction captures what verbose text looks like (function words
and filler, per the token heatmap and logit lens). It does not capture the decision to write more.

## Gemma 3 12B refusal benchmark

![Gemma 3 12B refusal rate and KL by variant](docs/figures/gemma-refusal.svg)

| variant | refusal rate, held-out (n=122) | KL |
|---|---|---|
| original | 91.8% | – |
| vectorlab trial 2 (ablate the layer-21 direction at layers 14–32), hooks | 8.2% | 0.199 |
| trial 2, weights | 9.0% | 0.196 |
| trial 2, weights with norms preserved | 9.0% | 0.195 |
| trial 29 (layer-19 direction at layers 0–46), weights | 6.6% | 0.106 |
| p-e-w/gemma-3-12b-it-heretic | 4.9% | 0.144 |

- **Same metric for every model.** All rows use the same refusal-phrase metric, prompts and KL reference.
  Heretic's model card reports 3/100 refusals and KL 0.16 by its own measure.
- **Remaining flags are caveats.** Most answers the edited models still get flagged for are mid-answer legal
  caveats, not refusals.
- **Massive-activation dimensions.** Gemma 3 has dimensions whose mean magnitude is over 100× the median;
  one of them (2339) carried up to 97% of the raw refusal direction. Ablating the raw direction wrecked the
  model (KL 10–30) instead of removing refusal. Ablation directions therefore leave those dimensions out.
- **Weights vs hooks.** Gemma's attention and MLP outputs pass through post-norms, so a weight edit cannot
  be an exact orthogonalization. vectorlab uses a ridge-regularized oblique projection through the post-norm
  scale instead. On gemma-3-4b-it, its KL to the hook ablation is 0.0037, against 0.088 for orthogonalizing
  against (1 + w) · r. On Qwen and Llama the weight edit is exact.

## Limitations

- **Gemma weight edits are approximations.** The oblique projection only approximately removes the direction
  after the post-norm. Additions cannot be baked into Gemma weights at all, and MLX export of a baked
  addition is unsupported for Qwen3, because mlx-lm has no attention bias.
- **Judges agree well but not perfectly.** κ is 0.90 (hedging) and 0.81 (sycophancy) against one labeler's
  60 labels. The sycophancy judge is +0.24 too lenient on unsteered baseline answers. It also rewarded at
  least one contrarian answer to a sound claim.
- **Synthetic datasets.** The hedging, sycophancy and verbosity pairs were written by the same 4B model that
  is being steered. Some sycophancy items have wrong premises or wrong "honest" answers. Hedging and
  sycophancy are largely separable by vocabulary alone.
- **Truncated answers.** Judged behaviors generate at most 256 tokens. Qwen3-4B's baseline answers to
  sycophancy prompts average 255 tokens, so the judge scores truncated answers, and the search's length
  penalty could not act.
- **Batch size changes greedy outputs.** In bf16, padding changes the numerics: the same hedging edit scores
  +0.38 at batch 32 and +0.28 at batch 16 on the same 60 prompts. That gap is about two-thirds of the CI
  width. `eval.json` and `compare.json` now record the batch size.
- **Phrase-based refusal metric.** It counts any refusal phrase anywhere in the answer, including a legal
  caveat inside an otherwise complete answer. It also never sees refusals phrased in other words.
- **Small samples, one seed.** Held-out sets are 60–122 prompts, every run uses split seed 0, and search
  numbers come from 32 train prompts.
