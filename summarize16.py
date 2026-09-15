"""Summarise v16_mc.json into the percentiles reported in Supplementary Table S14."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
src = HERE / 'v16_mc.json'
outfile = HERE / 'v16_mc_summary.json'
rows = json.loads(src.read_text())
if not rows:
    raise SystemExit('v16_mc.json is empty; run montecarlo16.py first')

summary = {}
keys = ['sq_delay','d_delay','e_delay','sq_min','d_min','e_min','sq_phi','d_phi','e_phi','d_meanU','e_meanU']
for key in keys:
    a = np.array([r[key] for r in rows], dtype=float)
    p5,p50,p95 = np.percentile(a,[5,50,95])
    summary[key] = {'p5':float(p5),'p50':float(p50),'p95':float(p95)}
summary['n'] = len(rows)
summary['dmpc_better'] = int(sum(r['d_delay'] < r['sq_delay'] for r in rows))
summary['screen_better'] = int(sum(r['e_delay'] < r['sq_delay'] for r in rows))
summary['nonconv'] = int(sum(r['nonconv'] for r in rows))
gain = np.array([r['sq_delay'] - r['d_delay'] for r in rows],dtype=float)
p5,p50,p95 = np.percentile(gain,[5,50,95])
summary['paired_delay_gain'] = {'p5':float(p5),'p50':float(p50),'p95':float(p95)}
outfile.write_text(json.dumps(summary,indent=1)+'\n')
print(json.dumps(summary,indent=2))
