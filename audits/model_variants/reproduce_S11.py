#!/usr/bin/env python3
"""Reproduce historical model-variant Table S15 and sparse order sweep in S11.

Legacy sequential control model is frozen, and its activation is overridden by
exactly the smooth v16 activation (epsilon=0.001). This does not change the
current common-price controller; its unchanged outputs are read from the
published v1.0.0 canonical JSON.

Usage: python audits/model_variants/reproduce_S11.py [--sparse-order-sweep] [--write]
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import legacy_sequential as GS

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

# Same differentiable threshold used in the nominal archived v16 pilot.
def smooth_activation(x,p=GS.NOMINAL):
    h=np.tanh(p.kappa*(x-p.theta))
    return 0.5*(h+np.sqrt(h*h+1.0e-6))
GS.activation=smooth_activation

# Sparse channel map: nine v14 retained entries, no refitting.
mask=np.zeros_like(GS.NOMINAL.Bu,dtype=bool)
for row,col in [(0,0),(7,0),(3,1),(4,2),(1,3),(2,3),(5,4),(6,4),(7,5)]:
    mask[row,col]=True
PARAMS={'dense_sequential':GS.NOMINAL,
        'sparse_sequential':replace(GS.NOMINAL,Bu=np.where(mask,GS.NOMINAL.Bu,0.0))}


def met(p,scenario,order=None):
    X,U,C=GS.simulate(scenario,p,order=order,diagnostics=False)
    z=GS.summarise(scenario,X,U,C,p)
    return {'delay':float(z['model_derived_diagnostic_delay_years']),
            'min_access':float(z['final_min_access']),
            'phi':float(z['final_phi']),
            'mean_intervention':float(z['mean_total_control_per_period']),
            'cumulative_actor_actions':[float(v) for v in U.sum(0)]}


def nominal():
    out={'source_tag':'v1.0.0', 'historical_legacy':'Gauss-Seidel residual-budget code from A2 audit',
         'activation':'smooth, epsilon=0.001', 'scenarios':{}}
    for name,p in PARAMS.items():
        out['scenarios'][name]={}
        for s in ('status_quo','dmpc','dmpc_equity'):
            out['scenarios'][name][s]=met(p,s)
    # Current published symmetric-price model is in the archive; avoid a second
    # implementation of its dynamics/coordination in this historical variant audit.
    main=json.loads((ROOT/'reference_results/v16_results_A.json').read_text())
    out['scenarios']['sparse_common_price']={}
    for label, key in [('status_quo','sq'),('dmpc','dmpc'),('dmpc_equity','screen')]:
        d=main[key]
        out['scenarios']['sparse_common_price'][label]={
            'delay':float(d['delay']),'min_access':float(d['minacc']),
            'phi':float(d['phi']),'mean_intervention':float(d['meanU']),
            'cumulative_actor_actions':[float(v) for v in d['cum']]}
    return out


def dense_order_spotcheck():
    """Non-exhaustive diagnostic: nominal and one cyclic-shift order on dense map."""
    orders=(tuple(range(GS.N_U)), tuple(range(1,GS.N_U))+ (0,))
    rows=[]
    for order in orders:
        d=met(PARAMS['dense_sequential'],'dmpc_equity',order)
        rows.append({'order':[GS.ACTOR_NAMES[i] for i in order],**d})
    return {'scope':'two selected dense-map orders only, not an exhaustive range',
            'orders':rows}

def check_dense(got):
    ref=json.loads((HERE/'reference_results/S11_dense_order_spotcheck.json').read_text())
    for v,w in zip(got['orders'],ref['orders']):
        assert v['order']==w['order']
        assert abs(v['delay']-w['delay'])<2e-4
        assert max(abs(a-b) for a,b in zip(v['cumulative_actor_actions'],w['cumulative_actor_actions']))<2e-3
    print('S11 dense-map spot check: two orders reproduced (non-exhaustive).')

def sparse_order_sweep():
    base=tuple(range(GS.N_U)); rev=tuple(reversed(base))
    orders=[tuple(seq[i:]+seq[:i]) for seq in (base,rev) for i in range(GS.N_U)]
    rows=[]
    for order in orders:
        d=met(PARAMS['sparse_sequential'],'dmpc_equity',order)
        rows.append({'order':[GS.ACTOR_NAMES[i] for i in order],**d})
        print('sparse order',len(rows),'of',len(orders),flush=True)
    c=np.array([r['cumulative_actor_actions'] for r in rows]);delay=np.array([r['delay'] for r in rows]);a=np.array([r['min_access'] for r in rows])
    return {'orders':rows,'delay_spread':float(np.ptp(delay)),
            'min_access_spread':float(np.ptp(a)),
            'cumulative_min':[float(v) for v in np.min(c,axis=0)],
            'cumulative_max':[float(v) for v in np.max(c,axis=0)]}


def check_nominal(got):
    ref=json.loads((HERE/'reference_results/S11_nominal_variants.json').read_text())
    for group in ref['scenarios']:
        for scenario in ref['scenarios'][group]:
            a=got['scenarios'][group][scenario];b=ref['scenarios'][group][scenario]
            for field in ('delay','min_access','phi'):
                assert abs(a[field]-b[field])<2.0e-4,(group,scenario,field,a[field],b[field])
    print('Table S15: 9 nominal scenario outcomes verified (including 3 archived common-price results).')


def check_orders(got):
    ref=json.loads((HERE/'reference_results/S11_sparse_order_sweep.json').read_text())
    assert len(got['orders'])==12
    for field in ('delay_spread','min_access_spread'):
        assert abs(got[field]-ref[field])<3e-4,(field,got[field],ref[field])
    for field in ('cumulative_min','cumulative_max'):
        assert max(abs(x-y) for x,y in zip(got[field],ref[field]))<3e-3,(field,got[field],ref[field])
    print('S11 sparse map: 12/12 tested actor update orders reproduced.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sparse-order-sweep',action='store_true',help='also regenerate the 12 sparse-map order tests (slower)')
    parser.add_argument('--dense-order-spotcheck',action='store_true',help='compare two dense update orders (non-exhaustive)')
    parser.add_argument('--write',action='store_true',help='overwrite canonical reference JSON (maintainers only)')
    args=parser.parse_args()
    got=nominal()
    if args.write:
        (HERE/'reference_results/S11_nominal_variants.json').write_text(json.dumps(got,indent=2)+'\n')
    else:check_nominal(got)
    if args.dense_order_spotcheck:
        dd=dense_order_spotcheck()
        if args.write:(HERE/'reference_results/S11_dense_order_spotcheck.json').write_text(json.dumps(dd,indent=2)+'\n')
        else:check_dense(dd)
    if args.sparse_order_sweep:
        rows=sparse_order_sweep()
        if args.write:(HERE/'reference_results/S11_sparse_order_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
        else:check_orders(rows)
