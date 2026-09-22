# C-VoC v0.5 empirical checkpoint — September 2026

**Provenance:** [GitHub Actions C-VoC Decision Research run](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35793045620), with downloadable full \`voc-576.json\` and \`voc-1152.json\` artifacts. This report is about finite, seeded **synthetic** contextual Bernoulli environments only. One workflow run and small toy distributions cannot establish general speed, safety or research novelty.

## Predeclared experiment

Three independently seeded families, 12 scenarios per family: crossed reward action curves, high-but-uncertain safety (near-safe), and safety probabilities close to the acceptance boundary. Each method sees the SAME potential-outcome stream for each (model, action) cell, with fixed external synthetic posterior per scenario. The analytic oracle knows every simulator probability and selects the highest expected reward that is feasible in all modeled regimes. Certified and exploratory outputs are evaluated separately; exploratory choices are NEVER certified.

Compare full-grid fixed (a higher-budget accuracy ceiling), equal-budget uniform, heuristic ACA, hybrid, exact reward-only one-batch EVSI, new joint reward+safety C-VoC, and an opt-in price-sensitive C-VoC governor. The fixed method uses 2304 cell draws for maximum=256; equal-budget policies use 1152. The governor's explicit reward-unit price was 0.0001 per draw and its stopping decision is myopic.

## 1152-query experiment: 36 scenarios

| Method | Mean draws | Certified oracle matches | Mean true constrained regret | Mean elapsed wall-time | Unsafe certified | Unsafe exploratory |
|---|---:|---:|---:|---:|---:|---:|
| Fixed full grid (2304) | 2304 | 18/36 | 0.05636 | 0.00678 s | 0 | 2 |
| Uniform | 1152 | 14/36 | 0.07476 | 0.00543 s | 0 | 8 |
| Legacy ACA | 1152 | 10/36 | 0.09254 | 0.01185 s | 0 | 2 |
| Hybrid | 1152 | 10/36 | 0.09254 | 0.01088 s | 0 | 2 |
| Exact reward-only EVSI | 1152 | 14/36 | 0.07476 | 0.03141 s | 0 | 8 |
| Joint reward+safety C-VoC | 1152 | 14/36 | 0.07493 | 0.01286 s | 0 | 8 |
| Opt-in C-VoC governor | 576 | 10/36 | 0.09254 | 0.00902 s | 0 | 4 |

Certified choices were analytically feasible in all 36 toy contexts in this finite run. **Do not infer a guarantee of safe external actions.** C-VoC's one-step joint value is strictly positive in a controlled unit test where reward-only EVSI is zero, confirming it *can* detect a safety-information opportunity under the specified model. Nevertheless, C-VoC did **not** outperform uniform on certified selection in the prespecified distribution. Its measured wall-time overhead was greater. The governor saved simulated draws but LOST certified decision accuracy compared with uniform in this run, and was still slower in measured elapsed time.

## Diagnostic insight

At small samples, a single batch of 16 safety observations is often insufficient to certify any non-baseline action under the uniform-over-time KL threshold. Thus the exact one-batch value is commonly zero even if several batches could eventually establish feasibility. The algorithm then falls back to uniform sampling, explaining similar selected actions. A governor that treats a zero one-batch gain as proof of zero *long-horizon* value can stop prematurely. The current governor intentionally requires explicit opt-in and a positive reward-unit compute price.

## Next falsifiable milestone

A multibatch, constrained value-of-information planner should explicitly model the option to invest successive queries until a safety certificate becomes possible, and compare expected regret reduction per **total measured CPU cost**. Require diverse independent holdout tasks and calibration under wrong priors, dependent reward/safety observations and nonstationary safety probabilities. Do not replace the baseline until evidence shows improved objective quality at matched computation cost. All positive and negative results are preserved in this repository.
