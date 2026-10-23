# Post-convergence unregularised best-response audit (v1.1.0)

This directory contains only code, machine-readable evidence and technical reproduction notes.
It is an **additive audit**, not a new controller or a numerical re-calibration.
Its input is the exact v1.0.0 `pilot16.py` source (Git blob
`680f138228b3f7464896d8377cede34178b7e55e`, commit
`b84219f6f338253b7fff58616d12acc9fad1aca8`).
The source is retained under `release_source/` to keep this audit tied to its
original, immutable mathematical specification.

## What is checked

For two 36-period nominal trajectories and six actors, the code records the full
4-step control predictions, prices and states, then solves two independent
**unregularised** problems per actor-period: (A) original actor cost plus frozen
common price over the box; (B) original actor cost with all other actors fixed
subject to the shared fiscal budget. The latter uses an audit-only projection
of small raw budget excesses; it does **not** alter nominal trajectories.
There are 432 actor-period cases and 864 response problems.

Each problem is searched from 26 SLSQP initialisations and two seeded differential-
evolution initialisations followed by polishing. The stored evidence includes
all responses, optimisers' statuses, residuals, nominal traces and checksums.
The 191 differential-evolution cap hits were re-run with the same seeds and an
extended cap; the completion evidence is stored in `results/`.

## Numerical summary (maxima of improvements *found*)

| Quantity | Plain | Shared access penalty |
|---|---:|---:|
| Priced objective reduction | 1.920879e-8 | 2.137313e-8 |
| Budget-feasible unilateral reduction (no price term) | 1.920879e-8 | 1.557542e-6 |
| Unit-step box projected-gradient residual at original candidate | 7.003165e-4 | 1.180435e-3 |
| Raw horizon budget excess | 0 | 4.834598e-7 |
| Complementarity product residual | 0 | 2.433512e-6 |

These are improvements **found by finite searches**, not certified upper bounds
on all possible deviations. In particular, convergence of the proximal
coordination iteration does not establish global variational equilibrium.

## Reproduction

The post-convergence audit was checked with Python 3.13.5, NumPy 2.3.5 and
SciPy 1.17.0, different from the original v1.0.0 simulation environment.
Use a **separate virtual environment**; do not change the original repository's
`requirements.txt` just for these audit scripts.

From this directory, after installing `requirements.txt`:

```bash
python verify_audit.py
```

A complete regeneration is more expensive and **overwrites evidence files**.
Use a disposable copy of this directory rather than a Git checkout if you run:

```bash
python reproduce_audit.py --workers 4
```

The orchestrator first validates and renames the delivered checksum list,
then performs the capture, searches, extended-cap completion, and verification.
`SOURCE_PROVENANCE.json` records the archival identity. No manuscript,
reviewer correspondence or third-party data files are included.
