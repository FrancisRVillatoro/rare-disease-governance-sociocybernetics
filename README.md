# Rare-disease governance socio-cybernetics — reproducibility code and data

Executable Python source, frozen machine-readable numerical results and technical
reproducibility notes supporting the computational illustrations associated with:

> **A socio-cybernetic architecture for rare-disease governance based on social systems theory**

**Manuscript authors (original order)**

- Enrique Fernández-Vilas — https://orcid.org/0000-0002-3107-6337
- Carolina Feliciana Machado — https://orcid.org/0000-0002-9685-1576
- Gema Esteban-Bueno — https://orcid.org/0000-0001-7554-9434
- Francisco R. Villatoro — https://orcid.org/0000-0003-4314-6213
- Juan Luis Fernández Martínez — https://orcid.org/0000-0002-4758-2832
- Juan R. Coca — https://orcid.org/0000-0003-1140-7351

**Scope:** only code, numeric data and technical reproduction documentation.
No manuscript, supplementary manuscript, PDF/Word of the article, editorial
correspondence, peer-review response or submission file is distributed here.
This is not an empirical policy controller, causal evaluation, clinical decision
system, or proof of existence/uniqueness of a nonlinear game equilibrium.

## Archived versions

- **v1.0.0 (16 September 2026):** frozen original pilot, comparator and sensitivity
  calculations. **Version DOI:** https://doi.org/10.5281/zenodo.22787139
- **v1.1.0 (release candidate, October 2026):** cumulative package, adding the unregularised
  post-convergence best-response numerical evidence and executable provenance
  for S6 and the S11 model-variant calculations. Its **version DOI is assigned
  by Zenodo after the v1.1.0 GitHub Release is published**. Do not use the
  v1.0.0 DOI to cite the additions in v1.1.0.
- **All-version concept DOI:** https://doi.org/10.5281/zenodo.22787138

## Contents

Existing original simulation workflows (unchanged from v1.0.0):

- `pilot16.py`: model, actor objectives and symmetric-price coordination.
- `audit16.py`: nominal comparators, deterministic sensitivity and diagnostics.
- `montecarlo16.py`, `summarize16.py`: 100 paired re-optimised stochastic runs.
- `figures16.py`: figure regeneration (figures generated locally, not archived as binaries).
- `reproduce_all.py`, `verify_release.py`: original orchestration and checks.
- `reference_results/`: frozen v1.0.0 numerical JSON outputs.

Additions in v1.1.0:

- `audits/unregularised_best_response/`: independent post-convergence unregularised
  objectives, 432 actor-period cases, 864 response problems, source trace and
  machine-readable detailed evidence. See the folder README for interpretation.
- `audits/model_variants/`: executable first-period convergence screen (S6),
  historical dense/sparse sequential model comparisons (S11/Table S15) and
  sparse-order sensitivity data, plus a non-exhaustive dense-order spot check. See the folder README.
- `RELEASE_NOTES_v1.1.0.md`, `SHA256SUMS_v1.1.0.txt`: release additions and hashes.

## Reproduction

The original model used Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1.
Matplotlib is needed for figures. In its original environment, run:

```bash
python reproduce_all.py --deterministic
python reproduce_all.py --full
```

For the S6 and S11 provenance scripts:

```bash
python audits/model_variants/reproduce_S6.py
python audits/model_variants/reproduce_S11.py
python audits/model_variants/reproduce_S11.py --sparse-order-sweep
python audits/model_variants/reproduce_S11.py --dense-order-spotcheck
```

For the new best-response audit, use a **separate environment** with
`audits/unregularised_best_response/requirements.txt` (Python 3.13.5,
NumPy 2.3.5, SciPy 1.17.0 were used for its checks). The stored audit can
be verified without re-running all stochastic searches:

```bash
python audits/unregularised_best_response/verify_audit.py
```

Complete regeneration (run from a disposable copy of that audit directory):

```bash
python reproduce_audit.py --workers 4
```

## Interpretation

Proximal-iteration stopping residuals, first-order stationarity diagnostics,
priced best-response checks and fiscally feasible unpriced deviations are
**different** numerical tests. No finite multi-start search certifies global
optimality or upper-bounds the worst possible unilateral improvement.
The shared access penalty is a substantive modelling assumption and does
not demonstrate redistribution or guaranteed floor satisfaction.

## Citation and licence

See `CITATION.cff` for the six creators and licence (MIT). For the original
v1.0.0 computations use DOI **10.5281/zenodo.22787139**. When citing the
new audit, use the v1.1.0 version DOI once Zenodo has issued it. The concept
DOI **10.5281/zenodo.22787138** represents the version family.
