"""Run the audit locally. Never writes to GitHub or Zenodo."""
from pathlib import Path
import argparse,os,subprocess,sys
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
if args.workers<1:raise SystemExit('--workers must be positive')
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
commands=[[sys.executable,'capture_nominal.py'],
          [sys.executable,'audit_best_responses.py','--workers',str(args.workers)],
          [sys.executable,'finish_diagnostics.py','--workers',str(args.workers)],
          [sys.executable,'verify_audit.py']]
# The hash manifest validates the delivered frozen package, not freshly overwritten
# evidence; validate it once, before changing any output files.
if (root/'SHA256SUMS.txt').exists():
    subprocess.run([sys.executable,'verify_audit.py'],cwd=root,env=env,check=True)
    (root/'SHA256SUMS.txt').rename(root/'SHA256SUMS_delivered.txt')
for command in commands:
    print('Executing:',' '.join(command),flush=True)
    subprocess.run(command,cwd=root,env=env,check=True)
