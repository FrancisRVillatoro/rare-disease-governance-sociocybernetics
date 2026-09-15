# montecarlo16.py: paired stochastic sensitivity with re-optimisation. Usage: python3 montecarlo16.py START END (appends runs START..END-1 to v16_mc.json; seeds 2026090600+r).
import numpy as np, json, time, sys
import pilot16 as P

START = int(sys.argv[1]); END = int(sys.argv[2])
BASE_SEED = 2026090600
import os
out = [] if START == 0 else (json.load(open('v16_mc.json')) if os.path.exists('v16_mc.json') else [])
t0 = time.time()
Ac_nom = P.Ac.copy(); Bu_nom = P.Bu.copy(); delta_nom = float(P.delta)
maskA = Ac_nom != 0; maskB = Bu_nom != 0
for r in range(START, END):
    rng = np.random.default_rng(BASE_SEED + r)
    Acp = Ac_nom.copy(); Acp[maskA] *= rng.uniform(0.9, 1.1, size=int(maskA.sum()))
    Bup = Bu_nom.copy(); Bup[maskB] *= rng.uniform(0.9, 1.1, size=int(maskB.sum()))
    dp = delta_nom * rng.uniform(0.9, 1.1)
    W = np.clip(rng.normal(0.0, 0.0015, size=(P.T, 8)), -0.0045, 0.0045)
    P.Ac = Acp; P.delta = dp
    m = P.Model(Bu=Bup)
    def run_sq():
        x = P.x0.copy(); X=[x.copy()]
        for t in range(P.T): x = m.step(x, P.uSQ, W[t]); X.append(x.copy())
        return np.array(X)
    Xs = run_sq()
    def run_cl(screen):
        x = P.x0.copy(); X=[x.copy()]; Uapp=[]; nonconv=0
        U = np.tile(P.uSQ[:,None],(1,m.Np)); lam = np.zeros(m.Np)
        for t in range(P.T):
            if t>0:
                U = np.concatenate([U[:,1:],U[:,-1:]],axis=1); lam = np.concatenate([lam[1:],lam[-1:]])
            U, lam, d = P.coordinate(x, U, lam, m, screen)
            nonconv += (not d['conv'])
            u = U[:,0].copy()
            if u.sum() > P.usum:
                if u.sum()-P.usum <= 2e-6: u = u*P.usum/u.sum(); U[:,0]=u
                else: nonconv += 1
            Uapp.append(u); x = m.step(x, u, W[t]); X.append(x.copy())
        return np.array(X), np.array(Uapp), nonconv
    Xd, Ud, ncd = run_cl(False)
    Xe, Ue, nce = run_cl(True)
    rec = dict(run=r, sq_delay=float(1/Xs[-1,0]), d_delay=float(1/Xd[-1,0]), e_delay=float(1/Xe[-1,0]),
               sq_min=float(P.access(Xs[-1]).min()), d_min=float(P.access(Xd[-1]).min()), e_min=float(P.access(Xe[-1]).min()),
               sq_phi=float(np.ptp(P.access(Xs[-1]))), d_phi=float(np.ptp(P.access(Xd[-1]))), e_phi=float(np.ptp(P.access(Xe[-1]))),
               d_meanU=float(Ud.sum(1).mean()), e_meanU=float(Ue.sum(1).mean()), nonconv=int(ncd+nce))
    out.append(rec)
    json.dump(out, open('v16_mc.json','w'), indent=0)
    print(r, rec, f'{time.time()-t0:.0f}s'); sys.stdout.flush()
