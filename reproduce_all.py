#!/usr/bin/env python3
"""Orchestrate the RDC v16 reproducibility workflow."""
from pathlib import Path
import argparse, subprocess, sys

HERE = Path(__file__).resolve().parent

def run(*args):
    print('+', ' '.join(map(str,args)), flush=True)
    subprocess.run([str(a) for a in args], cwd=HERE, check=True)

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--deterministic', action='store_true', help='nominal + deterministic audit/comparators only')
    g.add_argument('--full', action='store_true', help='also rerun all 100 stochastic replicates')
    args=ap.parse_args()
    py=sys.executable
    for name in ['v16_results_A.json','v16_results_B.json','v16_results_C.json','v16_mc.json','v16_mc_summary.json']:
        p=HERE/name
        if p.exists(): p.unlink()
    run(py,'pilot16.py')
    run(py,'audit16.py')
    if args.full:
        run(py,'montecarlo16.py','0','100')
        run(py,'summarize16.py')
        run(py,'figures16.py')
    run(py,'verify_release.py')
    print('Reproduction workflow completed successfully.')

if __name__=='__main__':
    main()
