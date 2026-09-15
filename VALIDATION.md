# Validation protocol

This repository is the frozen computational reproducibility object associated with the manuscript titled:

> **A socio-cybernetic architecture for rare-disease governance based on social systems theory**

## Deterministic calculations

The files `reference_results/v16_results_A.json`, `v16_results_B.json`, and `v16_results_C.json` are the canonical deterministic outputs. A clean run of `pilot16.py` followed by `audit16.py` should reproduce them within the tolerances encoded in `verify_release.py`.

## Paired stochastic sensitivity

Every stochastic replicate starts from the same frozen nominal `A_c`, `B_u`, and `delta`. Replicate-specific perturbations are applied independently; no perturbed parameter value is propagated to the next replicate. The paired status quo, symmetric-price DMPC and access-screen DMPC runs within each replicate use the same perturbed model and process-noise realisation.

The 100-run sample uses seeds `2026090600 + r` for `r=0,...,99`. The complete per-replicate record is regenerated as `v16_mc.json`; the frozen percentile summary is `reference_results/v16_mc_summary.json`.

## Verification

After a full reproduction run, `verify_release.py` compares the regenerated deterministic JSON outputs and the regenerated stochastic percentile summary with the frozen reference files using declared floating-point tolerances.

## Scope

These checks establish computational reproducibility of the stated stylised model only. They are not empirical validation, causal policy evaluation, proof of uniqueness of the nonlinear GNEP, or proof of convergence outside the reported runs.
