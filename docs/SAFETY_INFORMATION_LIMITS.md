# Why some safety decisions consume substantial simulation budgets

This note formalizes a **limited, model-conditional** information-theoretic fact. It does not prove the optimality of the SIL allocator or imply that entropy can determine outcomes.

## Sequential two-hypothesis identification bound

Consider two synthetic worlds that are identical except for a single queried (model, action) cell's iid Bernoulli safety probability: p>theta in one world, q<theta in the other. A sequential algorithm may adaptively query that cell or ANY OTHER cells. Let N be the number of safety draws from the distinguishing cell before it outputs a binary judgment: certified safe versus rejected unsafe. Suppose the output has both false-rejection and false-certification probabilities at most delta, with 0<delta<1/2, under both worlds. Assume the stopped likelihood ratio is integrable and E_p[N]<infinity.

Then

\[
\mathbb{E}_p[N]\ D_{\mathrm{Bern}}(p\Vert q)
\;\ge\; D_{\mathrm{Bern}}(1-\delta\Vert\delta).
\]

**Proof sketch.** Every other sampled cell has the same probability law in both worlds, so it contributes zero expected log-likelihood ratio between those worlds. A sample from the distinguishing cell contributes expected Bernoulli KL divergence D(p||q) under p. The adaptive chain rule for the stopped transcript and Wald's identity give transcript divergence E_p[N] D(p||q). Mapping the entire transcript to the final binary judgment cannot increase KL (data-processing inequality). For binary decision error <=delta under both worlds, divergence of the two output distributions is at least D(1-delta||delta). Combining proves the claim.

If q approaches theta from below, the denominator approaches D(p||theta). When p is very close to theta, this divergence becomes small and the necessary number of samples can be large. No adaptive allocator can learn a nearly indistinguishable safety condition from zero observations. This bound addresses **certifying that particular cell**, not the complexity of selecting a safe fallback without certifying alternatives.

## Implementation diagnostic and limitations

The terminal query result includes safety_information_requirements for each non-baseline action/model pair. The plug_in_samples_for_safety_certificate value is ceil(log(4*K*Nmax/delta)/D(phat||theta)) if phat>theta, and null otherwise. This is a PLUG-IN planning estimate at the observed empirical phat, neither an exact minimum nor a guarantee of success in that many samples. It can exceed the per-cell sample cap. A cap below this estimated requirement helps explain why an apparently valuable action remains uncertified. It does not justify relaxing the safety constraint.

The sequential lower bound above involves a two-world hypothesis test with controlled errors. It is mathematically distinct from the implementation's finite-grid union-bound KL confidence sequence and from its heuristic query priority. Real-world distributions need not be iid, Bernoulli, covered by the assumed model family or stationary.
