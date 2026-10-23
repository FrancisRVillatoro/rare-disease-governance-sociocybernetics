# Validation protocol

This repository is the frozen computational reproducibility object associated with the manuscript titled:

> **A socio-cybernetic architecture for rare-disease governance based on social systems theory**

## Deterministic calculations

The files `reference_results/v16_results_A.json`, `v16_results_B.json`, and `v16_results_C.json` are the canonical deterministic outputs. A clean run of `pilot16.py` followed by `audit16.py` should reproduce them within the tolerances encoded in `verify_release.py`.

## Paired stochastic sensitivity

Every stochastic replicate starts from the same frozen nominal `A_c`, `B_u`, and `delta`. Replicate-specific perturbations are applied independently; no perturbed parameter value is propagated to the next replicate. The paired status quo, symmetric-price DMPC and access-screen DMPC runs within each replicate use the same perturbed model and process-noise realisation.

The frozen 100-run sample uses seeds `2026090600 + r` for `r=0,...,99`. Full run records and percentile summaries are stored in `reference_results/v16_mc.json` and `reference_results/v16_mc_summary.json`.

## Static verification

After regenerating all outputs, run:

```bash
python verify_release.py
```

The verifier compares regenerated JSON results with the frozen reference files using declared floating-point tolerances.

## Scope

These checks establish computational reproducibility of the stated stylised model only. They are not empirical validation, causal policy evaluation, proof of uniqueness of the nonlinear GNEP, or proof of convergence outside the reported runs.

## v1.1.0 added numerical audits

The unregularised audit (in `audits/unregularised_best_response/`) was
independently reproduced against the tagged v1.0.0 source. Its stored output
has a verifier that recomputes 864 incumbent/best-response objective pairs and
checks feasibility, source identity, solver statuses and the delivered hashes.
The additional multi-start and differential-evolution searches report
**best improvements found**, not globally certified upper bounds.

`audits/model_variants/reproduce_S6.py` reruns the five first-period settings,
including one nonconverged parameter choice. `reproduce_S11.py` reproduces
historical dense/sparse sequential model outputs at the stated rounded
precision and optionally replays 12 sparse update orders or two specified
dense update orders. The dense spot check contradicts a previously
claimed general range of 0.05--0.30 for cumulative regulatory control;
it does not assert exhaustive dense-map order bounds.

The audit requirements differ from the original v1.0.0 solver requirements;
use separate virtual environments for exact environment reproduction.
