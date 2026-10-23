"""Verify integrity, source identity and the saved unregularised BR calculations."""
from pathlib import Path
import hashlib, json, math
import numpy as np
from audit_objective import LocalObjective, P
from audit_best_responses import project_feasible
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

def main():
    b=(ROOT/'release_source/pilot16.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    assert blob=='680f138228b3f7464896d8377cede34178b7e55e'
    summaries=json.loads((OUT/'nominal_replay.json').read_text())
    for v in summaries['runs'].values():
        assert max(v['published_summary_max_abs_differences'].values())<1e-5
        assert v['same_floor_crossing'] and v['local_failures']==0 and v['diag']['nonconv']==0
    rows=[json.loads(l) for l in (OUT/'best_response_details.jsonl').read_text().splitlines()]
    assert len(rows)==432
    assert len({(r['scenario'],r['t'],r['i']) for r in rows})==432
    traces={name:json.loads((OUT/f'trace_{name}.json').read_text()) for name in ['plain','screen']}
    max_value_error=0.;max_budget_violation=0.;call_count=0
    for r in rows:
        t=traces[r['scenario']]['periods'][r['t']]
        Ur=np.array(t['U_raw']);Uf=project_feasible(Ur);lam=np.array(t['lambda'])
        for kind in ['priced','feasible']:
            f=LocalObjective(r['i'],Ur if kind=='priced' else Uf,t['x'],r['scenario']=='screen',lam if kind=='priced' else None)
            rr=r[kind];u=np.array(rr['incumbent']);v=np.array(rr['best_response_found']);up=np.array(rr['upper_bounds'])
            assert (v>=-1e-14).all() and (v<=up+1e-14).all()
            fb=f.reference_value(u);fv=f.reference_value(v)
            err=max(abs(fb-rr['base_objective']),abs(fv-rr['best_objective_found']))
            max_value_error=max(max_value_error,err)
            assert err<5e-12
            assert abs(max(0,fb-fv)-rr['gain_found'])<5e-12
            assert len(rr['local_solves'])==26 and len(rr['de_solves'])==2
            call_count+=len(rr['local_solves'])+len(rr['de_solves'])+1
            assert rr['local_failures']==0 and rr['polish_failures']==0
            if kind=='feasible':
                UU=Uf.copy();UU[r['i']]=v
                max_budget_violation=max(max_budget_violation,float((UU.sum(0)-P.usum).max()))
                assert np.max(UU.sum(0)-P.usum)<1e-14
    ext=json.loads((OUT/'diagnostics_completion.json').read_text())
    assert ext['extended_cap_hits']==0 and ext['extended_polish_failures']==0
    assert ext['max_further_improvement']<1e-12
    hfile=ROOT/'SHA256SUMS.txt';checked=0
    if hfile.exists():
        for line in hfile.read_text().splitlines():
            checksum,name=line.split('  ',1)
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==checksum,name
            checked+=1
    print(json.dumps({'verified':True,'actor_period_cases':len(rows),'unregularised_problems':2*len(rows),
        'source_git_blob':blob,'saved_objective_max_abs_error':max_value_error,
        'feasible_deviation_max_budget_violation':max_budget_violation,
        'hashes_checked':checked,'scope':'verification of stored numerical evidence; not a global equilibrium certificate'},indent=2))
if __name__=='__main__':main()
