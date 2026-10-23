"""Complete DE searches that hit the first 300-generation cap; decompose gains.
The extension changes only an audit search budget, never the archived simulation.
Both the initial cap hits and extended results are retained and reported.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json, itertools, csv, time, argparse
import numpy as np
from scipy.optimize import differential_evolution, minimize
from audit_objective import LocalObjective, P
from audit_best_responses import project_feasible, enc
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'

def extend(task):
    scenario,t,i,kind,seed,x,U,lam,inc,up,oldbest=task
    U=np.array(U);inc=np.array(inc);up=np.array(up)
    fun=LocalObjective(i,U,x,scenario=='screen',lam if kind=='priced' else None)
    dr=differential_evolution(fun.batch,list(zip(np.zeros(P.Np),up)),
        strategy='best1bin',maxiter=1500,popsize=12,tol=1e-11,atol=1e-13,
        mutation=(.5,1.),recombination=.7,init='latinhypercube',polish=False,
        updating='deferred',vectorized=True,rng=np.random.default_rng(seed),x0=inc)
    u=np.clip(dr.x,0,up);value=fun.reference_value(u)
    pr=minimize(fun.value_grad,u,method='SLSQP',jac=True,bounds=list(zip(np.zeros(P.Np),up)),
                options={'ftol':1e-13,'maxiter':500})
    pu=np.clip(pr.x,0,up);pv=fun.reference_value(pu)
    return {'scenario':scenario,'t':t,'i':i,'kind':kind,'seed':seed,
            'initial_cap':300,'extended_cap':1500,'success':bool(dr.success),'nit':int(dr.nit),
            'message':str(dr.message),'value':value,'solution':u,'polish_success':bool(pr.success),
            'polish_value':pv,'polish_solution':pu,'previous_best':oldbest,
            'further_improvement_found':max(0.,oldbest-min(pv,value))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=4); args=ap.parse_args()
    entries=[json.loads(s) for s in (OUT/'best_response_details.jsonl').read_text().splitlines()]
    traces={name:json.loads((OUT/f'trace_{name}.json').read_text()) for name in ['plain','screen']}
    tasks=[];decomp=[]
    for r in entries:
        p=traces[r['scenario']]['periods'][r['t']];U=np.array(p['U_raw']);lam=np.array(p['lambda']);x=np.array(p['x'])
        Uf=project_feasible(U);i=r['i'];case=r['feasible'];u=np.array(case['incumbent']);v=np.array(case['best_response_found'])
        price_move=float(lam@(v-u));gain=case['base_objective']-case['best_objective_found']
        gap_l=gain-price_move;slack=P.usum-Uf.sum(0);bound=float(lam@slack)
        decomp.append({'scenario':r['scenario'],'t':r['t'],'actor':r['actor'],
            'unpriced_gain':gain,'same_profile_priced_gain':gap_l,'price_of_extra_resource':price_move,
            'price_times_available_slack_bound':bound,'identity_residual':abs(gain-gap_l-price_move),
            'max_abs_control_change':float(np.max(np.abs(v-u))),
            'objective_shift_due_to_feasible_profile_projection':case['incumbent_objective_shift_from_raw']})
        for kind in ['priced','feasible']:
            d=r[kind]
            for de in d['de_solves']:
                if not de['success']:
                    tasks.append((r['scenario'],r['t'],i,kind,de['seed'],x,U if kind=='priced' else Uf,
                                  lam,d['incumbent'],d['upper_bounds'],d['best_objective_found']))
    ext_settings={'initial_cap':300,'extended_cap':1500,'number_to_extend':len(tasks),
                  'reason':'Complete searches flagged by generation limit; no physical or coordinator parameters changed.',
                  'seeds':'Same per-problem DE seeds as initial audit','tolerances':'Unchanged: tol=1e-11, atol=1e-13'}
    (OUT/'extension_settings.json').write_text(json.dumps(ext_settings,indent=2)+'\n')
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        extended=list(ex.map(extend,tasks))
    (OUT/'de_cap_extensions.json').write_text(json.dumps(extended,default=enc,indent=2)+'\n')
    with (OUT/'feasible_gain_decomposition.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(decomp[0]));w.writeheader();w.writerows(decomp)
    stats={**ext_settings,'seconds':time.perf_counter()-start,
           'extended_success':sum(r['success'] for r in extended),
           'extended_cap_hits':sum(not r['success'] for r in extended),
           'max_generations':max(r['nit'] for r in extended),
           'extended_polish_failures':sum(not r['polish_success'] for r in extended),
           'max_further_improvement':max(r['further_improvement_found'] for r in extended),
           'gain_flags_over_1e_6':[d for d in decomp if d['unpriced_gain']>1e-6],
           'feasible_gain_decomposition':{}}
    for name in ['plain','screen']:
        rr=[d for d in decomp if d['scenario']==name]
        stats['feasible_gain_decomposition'][name]={
            'max_same_profile_priced_gain':max(d['same_profile_priced_gain'] for d in rr),
            'max_price_times_available_slack':max(d['price_times_available_slack_bound'] for d in rr),
            'max_incumbent_objective_shift_due_projection':max(abs(d['objective_shift_due_to_feasible_profile_projection']) for d in rr),
            'worst_gain':max(rr,key=lambda d:d['unpriced_gain'])}
    (OUT/'diagnostics_completion.json').write_text(json.dumps(stats,default=enc,indent=2)+'\n')
    print(json.dumps(stats,default=enc,indent=2),flush=True)
if __name__=='__main__':main()
