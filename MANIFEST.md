# Reproducibility manifest — v1.1.0

v1.1.0 is **cumulative**: all v1.0.0 source and canonical JSON results are
inherited unchanged. See `RELEASE_NOTES_v1.0.0.md` for the original release.

## v1.1.0 additions

- `audits/unregularised_best_response/` — scripts, source snapshot tied to the
  frozen Git blob, full nominal traces, 432 actor-period records, all solver
  results and aggregate summaries, technical README and file hashes.
- `audits/model_variants/reproduce_S6.py` — Table S6 convergence screen.
- `audits/model_variants/reproduce_S11.py` — historical S11/Table S15 variants and selective dense-order check.
- `audits/model_variants/legacy_sequential.py` — historical Gauss–Seidel model.
- `audits/model_variants/reference_results/` — frozen verification JSON files.
- `audits/model_variants/README.md` — provenance and reproduction instructions.
- `SHA256SUMS_v1.1.0.txt`, `RELEASE_NOTES_v1.1.0.md`, updated README,
  VALIDATION and CITATION metadata.

Existing original scripts `pilot16.py`, `audit16.py`, `montecarlo16.py`,
`summarize16.py`, `figures16.py`, `reproduce_all.py` and `verify_release.py`
are not replaced. Existing original `reference_results/` are not replaced.

No article/manuscript PDF, Word, LaTeX, reviewer correspondence or proprietary
source file is part of the version-controlled reproducibility archive.
