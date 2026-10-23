"""Unregularised priced and feasible-unilateral BR audits of archived nominal runs.

Run capture_nominal.py first. This script never updates a nominal trajectory,
model parameter, coordinator tolerance, repository or Zenodo record.
"""
from pathlib import Path
import argparse, csv, itertools, json, time, platform, hashlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy
from scipy.optimize import minimize, differential_evolution
from audit_objective import LocalObjective, P
ROOT=Path(__file__).resolve().parent
ACTORS=['H','R','HTA','B','V','O']
CONFIG={
    'seed_base':2026102000,'horizon':4,'decision_periods':36,'scenarios':['plain','screen'],
    'tests':['priced_box_unregularised','feasible_unilateral_unpriced'],
    'multistarts':{'incumbent':1,'midpoint':1,'all_box_vertices':16,'uniform_random':8},
    'slsqp':{'maxiter':500,'ftol':1e-13,'jac':'independent_analytic_forward_sensitivities'},
    'differential_evolution':{'seeds_per_problem':2,'maxiter':300,'popsize':12,'tol':1e-11,
        'atol':1e-13,'mutation':[.5,1.],'recombination':.7,'strategy':'best1bin',
        'init':'latinhypercube','polish':False,'updating':'deferred','vectorized':True},
    'original_objective_slsqp_crosscheck':True,
    'material_gain_flag_absolute':1e-6,
    'projected_stationarity_step':1.0,
    'feasible_incumbent':'proportional columnwise downward projection only in audit; original controls unchanged',
    'scope':'all actors and all prediction horizons along the two nominal 36-period trajectories; no stochastic audit',
    'equilibrium_claim':'no global optimality, uniqueness or exact variational equilibrium certified'
}

def enc(o):
    if isinstance(o,np.ndarray):return o.tolist()
    if isinstance(o,np.generic):return o.item()
    raise TypeError(type(o).__name__)

def project_feasible(U):
    V=U.copy();b=V.sum(0)
    for k in range(P.Np):
        if b[k]>P.usum:
            V[:,k]*=np.nextafter(P.usum/b[k],0.)
    assert np.max(V.sum(0)-P.usum)<5e-16
    return V

def grad_check(fun,ui):
    f,g=fun.value_grad(ui);errs=[]
    for h in (1e-6,3e-7):
        fd=[]
        for k in range(P.Np):
            d=np.zeros(P.Np);d[k]=h
            fd.append((fun.reference_value(ui+d)-fun.reference_value(ui-d))/(2*h))
        errs.append(float(np.max(np.abs(np.array(fd)-g))))
    return min(errs)

def solve_one(fun,inc,upper,seed):
    inc=np.minimum(np.maximum(inc,0.),upper)
    rng=np.random.default_rng(seed)
    starts=[('incumbent',inc),('midpoint',upper/2)]
    for k,v in enumerate(itertools.product([0.,1.],repeat=P.Np)):
        starts.append((f'vertex_{k}',upper*np.array(v)))
    for k in range(8):starts.append((f'random_{k}',rng.uniform(size=P.Np)*upper))
    candidates=[('incumbent',inc,fun.reference_value(inc))];logs=[]
    for label,u in starts:
        res=minimize(fun.value_grad,u,method='SLSQP',jac=True,
                     bounds=list(zip(np.zeros(P.Np),upper)),options={'ftol':1e-13,'maxiter':500})
        uu=np.clip(res.x,0.,upper);v=fun.reference_value(uu)
        candidates.append((label,uu,v))
        logs.append({'method':'slsqp_analytic','start':label,'initial':u,'solution':uu,
            'value':v,'success':bool(res.success),'status':int(res.status),
            'nit':int(getattr(res,'nit',0)),'message':str(res.message)})
    de_logs=[]
    for j in range(2):
        dr=differential_evolution(fun.batch,list(zip(np.zeros(P.Np),upper)),
            strategy='best1bin',maxiter=300,popsize=12,tol=1e-11,atol=1e-13,
            mutation=(.5,1.),recombination=.7,init='latinhypercube',
            polish=False,updating='deferred',vectorized=True,
            rng=np.random.default_rng(seed+10000+j),x0=inc)
        du=np.clip(dr.x,0.,upper);dv=fun.reference_value(du)
        candidates.append((f'de_{j}',du,dv))
        polish=minimize(fun.value_grad,du,method='SLSQP',jac=True,
            bounds=list(zip(np.zeros(P.Np),upper)),options={'ftol':1e-13,'maxiter':500})
        pu=np.clip(polish.x,0.,upper);pv=fun.reference_value(pu)
        candidates.append((f'de_{j}_polished',pu,pv))
        de_logs.append({'seed':seed+10000+j,'success':bool(dr.success),
            'message':str(dr.message),'nit':int(dr.nit),'nfev_batch':int(dr.nfev),
            'solution':du,'value':dv,'polish_solution':pu,'polish_value':pv,
            'polish_success':bool(polish.success),'polish_status':int(polish.status)})
    _,bestu,bestv=min(candidates,key=lambda x:x[2])
    fd=minimize(fun.reference_value,bestu,method='SLSQP',
                bounds=list(zip(np.zeros(P.Np),upper)),options={'ftol':1e-13,'maxiter':500})
    fdu=np.clip(fd.x,0.,upper);fdv=fun.reference_value(fdu)
    candidates.append(('original_objective_fd',fdu,fdv))
    bestlabel,bestu,bestv=min(candidates,key=lambda x:x[2])
    base=fun.reference_value(inc)
    base_grad=fun.value_grad(inc)[1];opt_grad=fun.value_grad(bestu)[1]
    def pg(u,g):return float(np.max(np.abs(u-np.clip(u-g,0.,upper))))
    vals=np.array([l['value'] for l in logs])
    return {'base_objective':base,'best_objective_found':bestv,'gain_found':max(0.,base-bestv),
            'relative_gain_found':max(0.,base-bestv)/max(1.,abs(base)),
            'best_method':bestlabel,'incumbent':inc,'best_response_found':bestu,
            'max_control_change':float(np.max(np.abs(bestu-inc))),
            'projected_gradient_incumbent':pg(inc,base_grad),
            'projected_gradient_at_best_found':pg(bestu,opt_grad),
            'base_gradient':base_grad,'upper_bounds':upper,
            'multistart_objective_spread':float(np.ptp(vals)),
            'local_failures':sum(not l['success'] for l in logs),
            'de_failures':sum(not l['success'] for l in de_logs),
            'polish_failures':sum(not l['polish_success'] for l in de_logs),
            'fd_crosscheck':{'success':bool(fd.success),'status':int(fd.status),
                            'value':fdv,'solution':fdu,'message':str(fd.message)},
            'local_solves':logs,'de_solves':de_logs}

def worker(task):
    scenario,t,i,x,U,lam=task
    U=np.array(U);x=np.array(x);lam=np.array(lam);screen=scenario=='screen'
    seed=CONFIG['seed_base']+(1000000 if screen else 0)+1000*t+10*i
    Uf=project_feasible(U)
    funs=[('priced',LocalObjective(i,U,x,screen,lam),U[i],np.full(P.Np,P.umax)),
          ('feasible',LocalObjective(i,Uf,x,screen),Uf[i],
           np.minimum(P.umax,np.maximum(0.,P.usum-(Uf.sum(0)-Uf[i]))))]
    ans={'scenario':scenario,'t':t,'actor':ACTORS[i],'i':i,
         'audit_projection_max_change':float(np.max(np.abs(Uf-U))),
         'raw_budget_violation':float(np.maximum(U.sum(0)-P.usum,0.).max()),
         'raw_complementarity':float(np.max(np.abs(lam*(U.sum(0)-P.usum)))),
         'feasible_complementarity':float(np.max(np.abs(lam*(Uf.sum(0)-P.usum))))}
    for k,(kind,fun,inc,up) in enumerate(funs):
        rng=np.random.default_rng(seed+50000+k)
        testu=np.vstack([inc,up,np.zeros(P.Np),rng.uniform(size=(5,P.Np))*up])
        ref=np.array([fun.reference_value(u) for u in testu])
        bat=fun.batch(testu.T);ana=np.array([fun.value_grad(u)[0] for u in testu])
        ve=float(max(np.max(np.abs(ref-bat)),np.max(np.abs(ref-ana))))
        ge=max(grad_check(fun,inc),grad_check(fun,up/2))
        if ve>5e-12 or ge>2e-6:raise RuntimeError(f'Objective/gradient validation failure: {scenario,t,i,kind,ve,ge}')
        ans[kind]=solve_one(fun,inc,up,seed+k)
        ans[kind]['objective_validation_max_abs']=ve
        ans[kind]['gradient_validation_max_abs']=ge
        paths=np.vstack([fun.predicted_states(u) for u in testu])
        ans[kind]['validation_prediction_min']=float(paths.min())
        ans[kind]['validation_prediction_max']=float(paths.max())
        ans[kind]['incumbent_objective_shift_from_raw']=float(
            P.J_local(Uf[i],i,Uf,x,P.Model(),screen)-P.J_local(U[i],i,U,x,P.Model(),screen)) if kind=='feasible' else 0.
    return ans

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'audit_settings.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    tasks=[]
    for name in ['plain','screen']:
        d=json.loads((out/f'trace_{name}.json').read_text())
        for p in d['periods']:
            for i in range(P.Na):tasks.append((name,p['t'],i,p['x'],p['U_raw'],p['lambda']))
    if args.limit:tasks=tasks[:args.limit]
    started=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex, (out/'best_response_details.jsonl').open('w') as f:
        for n,r in enumerate(ex.map(worker,tasks),1):
            records.append(r);f.write(json.dumps(r,default=enc)+'\n');f.flush()
            if n%12==0 or n==len(tasks):
                print(f'{n}/{len(tasks)}: {r["scenario"]}, t={r["t"]}; elapsed={time.perf_counter()-started:.1f}s; max gains priced={max(s["priced"]["gain_found"] for s in records):.3e}, feasible={max(s["feasible"]["gain_found"] for s in records):.3e}',flush=True)
    summary={'settings':CONFIG,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
             'runtime_seconds':time.perf_counter()-started,'actor_period_cases':len(records),'problem_count':2*len(records),'scenarios':{}}
    rows=[]
    for a in records:
        row={k:a[k] for k in ['scenario','t','actor','audit_projection_max_change','raw_budget_violation','raw_complementarity','feasible_complementarity']}
        for kind in ['priced','feasible']:
            for k in ['base_objective','best_objective_found','gain_found','relative_gain_found',
                      'max_control_change','projected_gradient_incumbent','projected_gradient_at_best_found',
                      'multistart_objective_spread','local_failures','de_failures','polish_failures',
                      'objective_validation_max_abs','gradient_validation_max_abs','incumbent_objective_shift_from_raw']:
                row[f'{kind}_{k}']=a[kind][k]
            row[kind+'_fd_failure']=not a[kind]['fd_crosscheck']['success']
        rows.append(row)
    with (out/'best_response_by_actor_period.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name in ['plain','screen']:
        rec=[a for a in records if a['scenario']==name]
        if not rec:continue
        sn={'cases':len(rec),'raw_budget_violation_max':max(a['raw_budget_violation'] for a in rec),
            'raw_complementarity_max':max(a['raw_complementarity'] for a in rec),
            'audit_projection_max':max(a['audit_projection_max_change'] for a in rec)}
        for kind in ['priced','feasible']:
            keys=['gain_found','relative_gain_found','max_control_change','projected_gradient_incumbent',
                  'projected_gradient_at_best_found','multistart_objective_spread',
                  'objective_validation_max_abs','gradient_validation_max_abs']
            sk={k+'_max':max(a[kind][k] for a in rec) for k in keys}
            for k in ['local_failures','de_failures','polish_failures']:sk[k]=sum(a[kind][k] for a in rec)
            sk['fd_failures']=sum(not a[kind]['fd_crosscheck']['success'] for a in rec)
            sk['cases_above_material_gain_flag']=sum(a[kind]['gain_found']>CONFIG['material_gain_flag_absolute'] for a in rec)
            worst=max(rec,key=lambda a:a[kind]['gain_found'])
            sk['worst_gain_case']={'t':worst['t'],'actor':worst['actor'],'base':worst[kind]['base_objective'],'gain':worst[kind]['gain_found']}
            sn[kind]=sk
        summary['scenarios'][name]=sn
    (out/'best_response_summary.json').write_text(json.dumps(summary,default=enc,indent=2)+'\n')
    print(json.dumps(summary,default=enc,indent=2),flush=True)
if __name__=='__main__':main()
