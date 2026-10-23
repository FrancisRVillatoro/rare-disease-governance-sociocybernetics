"""Capture all horizons and multipliers without changing the published coordinator.
Uses byte-identical v1.0.0 pilot16.py. Wrappers only copy returned arrays and
record OptimizeResult statuses. No optimiser option, parameter or action changes.
"""
from pathlib import Path
import sys, json, hashlib, time, platform
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'release_source'))
import pilot16 as P

def encode(o):
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,np.generic): return o.item()
    raise TypeError(type(o).__name__)

def main():
    source=(ROOT/'release_source/pilot16.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
    assert blob=='680f138228b3f7464896d8377cede34178b7e55e',blob
    out=ROOT/'results';out.mkdir(exist_ok=True)
    reference=json.loads((ROOT/'release_source/reference_results/v16_results_A.json').read_text())
    meta={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'commit':'b84219f6f338253b7fff58616d12acc9fad1aca8','tag':'v1.0.0',
          'pilot_git_blob_sha1':blob,'pilot_sha256':hashlib.sha256(source).hexdigest(),'runs':{}}
    orig_coordinate=P.coordinate;orig_minimise=P.minimize
    for screen,name in [(False,'plain'),(True,'screen')]:
        records=[];solver_records=[]
        def logged_minimise(*args,**kwargs):
            res=orig_minimise(*args,**kwargs)
            solver_records.append({'success':bool(res.success),'status':int(res.status),
                                    'nit':int(res.nit),'message':str(res.message)})
            return res
        def logged_coordinate(x,U,lam,m,screen):
            t=len(records);n0=len(solver_records)
            UU,ll,dd=orig_coordinate(x,U,lam,m,screen)
            records.append({'t':t,'x':x.copy(),'U_raw':UU.copy(),'lambda':ll.copy(),
                            'coordination':dd.copy(),'local_calls':len(solver_records)-n0,
                            'local_failures':sum(not a['success'] for a in solver_records[n0:])})
            return UU,ll,dd
        P.coordinate=logged_coordinate;P.minimize=logged_minimise
        start=time.perf_counter()
        X,controls,prices,diag=P.run_dmpc(P.Model(),screen)
        elapsed=time.perf_counter()-start
        P.coordinate=orig_coordinate;P.minimize=orig_minimise
        a=np.array([P.access(x) for x in X]);tot=controls.sum(1)
        measured={'delay':1/X[-1,0],'minacc':a[-1].min(),'phi':np.ptp(a[-1]),
                  'meanU':tot.mean(),'maxU':tot.max(),'first':next((int(t) for t in range(len(X)) if a[t].min()>=P.abar),None),
                  'J':P.realised_cost(P.Model(),X,controls),'xT':X[-1].tolist(),'cum':controls.sum(0).tolist()}
        ref=reference['screen' if screen else 'dmpc']
        differences={k:float(np.max(np.abs(np.asarray(measured[k])-np.asarray(ref[k]))))
                     for k in measured if k!='first'}
        record={'scenario':name,'screen':screen,'x':X,'controls':controls,'first_prices':prices,
                'diag':diag,'periods':records,'local_solver_results':solver_records}
        (out/f'trace_{name}.json').write_text(json.dumps(record,default=encode,indent=2)+'\n')
        meta['runs'][name]={'seconds':elapsed,'summary':measured,'diag':diag,
                            'local_calls':len(solver_records),'local_failures':sum(not a['success'] for a in solver_records),
                            'published_summary_max_abs_differences':differences,
                            'same_floor_crossing':measured['first']==ref['first']}
        print(name,json.dumps(meta['runs'][name],default=encode),flush=True)
        (out/'nominal_replay.json').write_text(json.dumps(meta,default=encode,indent=2)+'\n')
if __name__=='__main__': main()
