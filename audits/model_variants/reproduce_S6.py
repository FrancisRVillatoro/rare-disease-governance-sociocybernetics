#!/usr/bin/env python3
"""Reproduce Supplementary Table S6 from the frozen v1.0.0 controller.

No parameter retuning, no policy-outcome selection. First horizon only.
Usage: python audits/model_variants/reproduce_S6.py [--write]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pilot16 as P

HERE = Path(__file__).resolve().parent
SETTINGS = [(20.,200.,.6),(20.,20.,.5),(20.,50.,.5),(40.,20.,.5),(80.,20.,.5)]


def run():
    records=[]
    original=(P.tau,P.gamma,P.omega)
    try:
        for tau,gamma,omega in SETTINGS:
            P.tau,P.gamma,P.omega=tau,gamma,omega
            U0=np.tile(P.uSQ[:,None],(1,P.Np))
            U,lam,diag=P.coordinate(P.x0.copy(),U0,np.zeros(P.Np),P.Model(),True)
            records.append({'tau':tau,'gamma':gamma,'omega':omega,
               'iterations':int(diag['it']),'converged':bool(diag['conv']),
               'budget_residual':float(diag['rp']),
               'control_change':float(diag['rU']),
               'dual_change':float(diag['rl']),
               'first_action':[float(v) for v in U[:,0]],
               'first_price':float(lam[0])})
    finally:
        P.tau,P.gamma,P.omega=original
    return {'source_tag':'v1.0.0','source_commit':'b84219f6f338253b7fff58616d12acc9fad1aca8',
            'experiment':'first-period proximal-Jacobi convergence screen (Supplementary Table S6)',
            'scenarios':records}


def check(generated):
    stored=json.loads((HERE/'reference_results/S6_convergence_screen.json').read_text())
    for got,ref in zip(generated['scenarios'],stored['scenarios']):
        for field in ('tau','gamma','omega','iterations','converged'):
            assert got[field]==ref[field],(field,got[field],ref[field])
        # Numerical residuals can vary slightly with SciPy and floating point.
        for field in ('budget_residual','control_change','dual_change'):
            assert abs(got[field]-ref[field]) < max(2e-5,0.03*abs(ref[field])),(field,got[field],ref[field])
    print('Table S6: 5/5 convergence-screen settings reproduced.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true',help='regenerate frozen JSON (maintainers only)');args=ap.parse_args()
    out=run()
    if args.write:
        p=HERE/'reference_results/S6_convergence_screen.json'
        p.write_text(json.dumps(out,indent=2)+'\n')
        print('Wrote',p)
    else:check(out)
    print(json.dumps([{'tau':v['tau'],'gamma':v['gamma'],'omega':v['omega'],'iterations':v['iterations'],'converged':v['converged']} for v in out['scenarios']],indent=2))
