"""Compare regenerated canonical summaries with the frozen reference outputs."""
from pathlib import Path
import json, math, sys

HERE=Path(__file__).resolve().parent
REF=HERE/'reference_results'
RTOL=5e-6; ATOL=5e-8
errors=[]

def cmp(a,b,path):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b): errors.append(f'{path}: keys differ: {set(a)^set(b)}')
        for k in set(a)&set(b): cmp(a[k],b[k],f'{path}/{k}')
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): errors.append(f'{path}: lengths {len(a)} != {len(b)}')
        for i,(x,y) in enumerate(zip(a,b)): cmp(x,y,f'{path}[{i}]')
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if not math.isclose(float(a),float(b),rel_tol=RTOL,abs_tol=ATOL): errors.append(f'{path}: {a} != {b}')
    elif a!=b:
        errors.append(f'{path}: {a!r} != {b!r}')

pairs=[('v16_results_A.json','v16_results_A.json'),('v16_results_B.json','v16_results_B.json'),('v16_results_C.json','v16_results_C.json')]
if (HERE/'v16_mc_summary.json').exists():
    pairs.append(('v16_mc_summary.json','v16_mc_summary.json'))
for gen_name,ref_name in pairs:
    gen=HERE/gen_name; ref=REF/ref_name
    if not gen.exists(): errors.append(f'missing generated {gen_name}'); continue
    if not ref.exists(): errors.append(f'missing reference {ref_name}'); continue
    cmp(json.loads(gen.read_text()),json.loads(ref.read_text()),gen_name)
if errors:
    print('VERIFY FAIL')
    for e in errors[:100]: print(' -',e)
    if len(errors)>100: print(f' ... {len(errors)-100} more')
    sys.exit(1)
print('VERIFY PASS: regenerated canonical summaries match frozen reference values within declared tolerances.')
