# audit16.py: nominal runs, comparators, per-actor costs, deterministic sensitivity (both variants), rho_a and epsilon sensitivity, algorithm-parameter, initialisation and label-permutation tests, autonomous diagnostic and long run. Run after pilot16.py in the same directory. Writes v16_results_A/B/C.json.
import numpy as np, json, time, sys
from scipy.optimize import minimize
import pilot16 as P

names = ['H','R','HTA','B','V','O']
R = {}
def met(m, X, U):
    a = np.array([P.access(x) for x in X]); amin = a.min(axis=1); phi = a.max(axis=1)-a.min(axis=1)
    first = next((t for t in range(len(X)) if amin[t] >= 0.40), None)
    return dict(delay=float(1/X[-1,0]), minacc=float(amin[-1]), phi=float(phi[-1]), meanU=float(U.sum(axis=1).mean()),
                maxU=float(U.sum(axis=1).max()), first=first, J=float(P.realised_cost(m,X,U)), xT=[float(v) for v in X[-1]],
                cum=[float(v) for v in U.sum(axis=0)])

m = P.Model()
t0 = time.time()
Xs,Us = P.run_const(m, P.uSQ); R['sq'] = met(m,Xs,Us)
Xd,Ud,ld,dd = P.run_dmpc(m, False); R['dmpc'] = met(m,Xd,Ud); R['dmpc']['diag'] = {k:float(v) for k,v in dd.items()}
Xe,Ue,le,de = P.run_dmpc(m, True); R['screen'] = met(m,Xe,Ue); R['screen']['diag'] = {k:float(v) for k,v in de.items()}
R['screen']['lambda0'] = [float(v) for v in le]; R['screen']['Jscreen_incl'] = float(P.realised_cost(m,Xe,Ue,True))
R['screen']['u_first8'] = [[float(v) for v in row] for row in Ue[:8]]
print('nominal done', time.time()-t0); sys.stdout.flush()
np.save('v16_Xs.npy',Xs); np.save('v16_Xd.npy',Xd); np.save('v16_Ud.npy',Ud); np.save('v16_Xe.npy',Xe); np.save('v16_Ue.npy',Ue); np.save('v16_le.npy',le)
spend = Ud.sum()/P.T; R['dmpc_spend'] = float(spend)
X,U = P.run_const(m, P.uSQ*spend/P.uSQ.sum()); R['sq_scaled'] = met(m,X,U); np.save('v16_Xsqs.npy',X)
def best_const(total):
    def obj(u):
        X,U = P.run_const(m,u); return P.realised_cost(m,X,U)
    cons=[{'type':'eq','fun':lambda u: np.sum(u)-total}]; best=None
    for guess in ([total/6]*6, [min(0.08,total),0,0,max(0,total-0.08),0,0], [0.04,0.01,0.01,0.02,0.01,0.01]):
        g=np.clip(np.array(guess,float),0,0.08); g*=total/g.sum()
        res=minimize(obj,g,method='SLSQP',bounds=[(0,0.08)]*6,constraints=cons,options={'maxiter':300,'ftol':1e-12})
        if best is None or res.fun<best.fun: best=res
    return best
bc=best_const(spend); X,U=P.run_const(m,bc.x); R['const_equal']=met(m,X,U); R['const_equal']['u']=[float(v) for v in bc.x]; np.save('v16_Xce.npy',X)
bf=best_const(0.18); X,U=P.run_const(m,bf.x); R['const_full']=met(m,X,U); R['const_full']['u']=[float(v) for v in bf.x]; np.save('v16_Xcf.npy',X)
def central_run(screen=False):
    x=P.x0.copy(); X=[x.copy()]; Uapp=[]; U=np.tile(P.uSQ[:,None],(1,m.Np))
    for t in range(P.T):
        if t>0: U=np.concatenate([U[:,1:],U[:,-1:]],axis=1)
        def obj(Uf):
            UU=Uf.reshape(6,m.Np); return sum(P.J_local(UU[i],i,UU,x,m,screen) for i in range(6))
        cons=[{'type':'ineq','fun':(lambda Uf,k=k: 0.18-np.sum(Uf.reshape(6,m.Np)[:,k]))} for k in range(m.Np)]
        res=minimize(obj,U.flatten(),method='SLSQP',bounds=[(0,0.08)]*(6*m.Np),constraints=cons,options={'maxiter':300,'ftol':1e-10})
        U=res.x.reshape(6,m.Np); u=U[:,0].copy(); Uapp.append(u); x=m.step(x,u); X.append(x.copy())
    return np.array(X),np.array(Uapp)
Xc,Uc=central_run(); R['central']=met(m,Xc,Uc); np.save('v16_Xc.npy',Xc)
def per_actor(X,U):
    out=np.zeros(6)
    for t in range(len(U)):
        e=X[t+1]-m.xstar
        for i in range(6): out[i]+=np.sum(m.Q[i]*e*e)+m.r[i]*U[t,i]**2
    return out
R['per_actor']={'dmpc':[float(v) for v in per_actor(Xd,Ud)],'central':[float(v) for v in per_actor(Xc,Uc)],'screen':[float(v) for v in per_actor(Xe,Ue)]}
R['gap']=float(R['dmpc']['J']/R['central']['J']-1)
x=P.x0; pen=400*np.sum(np.maximum(0,0.40-P.access(x))**2)
R['screen_pen_x0']=float(pen); R['tracking_x0']=[float(np.sum(m.Q[i]*(x-m.xstar)**2)) for i in range(6)]
print('comparators done', time.time()-t0); sys.stdout.flush()
json.dump(R, open('v16_results_A.json','w'), indent=1)

S={}
for lab,mm in [('Np2',P.Model(Np=2)),('Np6',P.Model(Np=6)),('Np8',P.Model(Np=8)),('r05',P.Model(r=P.r*0.5)),('r2',P.Model(r=P.r*2)),
               ('L05',P.Model(Lam=P.Lam*0.5)),('L2',P.Model(Lam=P.Lam*2)),('xmod',P.Model(xstar=P.x0+0.5*(P.xstar-P.x0)))]:
    X,U,l,d=P.run_dmpc(mm,False); S[lab]={'plain':met(mm,X,U)}; S[lab]['plain']['lam_pos']=int(np.sum(l>1e-9)); S[lab]['plain']['maxit']=int(d['max_it']); S[lab]['plain']['nonconv']=int(d['nonconv'])
    X,U,l,d=P.run_dmpc(mm,True); S[lab]['screen']=met(mm,X,U); S[lab]['screen']['lam_pos']=int(np.sum(l>1e-9)); S[lab]['screen']['maxit']=int(d['max_it']); S[lab]['screen']['nonconv']=int(d['nonconv'])
    print(lab, 'done', time.time()-t0); sys.stdout.flush()
    json.dump(S, open('v16_results_B.json','w'), indent=1)

R={}
t0=time.time()
def met(m,X,U):
    a=np.array([P.access(x) for x in X]); amin=a.min(axis=1); phi=a.max(axis=1)-a.min(axis=1)
    first=next((t for t in range(len(X)) if amin[t]>=0.40), None)
    return dict(delay=float(1/X[-1,0]),minacc=float(amin[-1]),phi=float(phi[-1]),meanU=float(U.sum(axis=1).mean()),maxU=float(U.sum(axis=1).max()),first=first,J=float(P.realised_cost(m,X,U)),cum=[float(v) for v in U.sum(axis=0)])
m=P.Model()
Ue=np.load('v16_Ue.npy'); Xe=np.load('v16_Xe.npy')
R['rho_a']={}
for rho in (100.0,1600.0):
    P.rho_a=rho
    X,U,l,d=P.run_dmpc(m,True); R['rho_a'][str(int(rho))]=met(m,X,U); R['rho_a'][str(int(rho))].update(lam_pos=int(np.sum(l>1e-9)),lam_max=float(l.max()),maxit=int(d['max_it']),nonconv=int(d['nonconv']),u_first3=[[float(v) for v in row] for row in U[:3]])
    print('rho',rho,'done',time.time()-t0); sys.stdout.flush()
P.rho_a=400.0
R['eps']={}
for eps in (1e-4,1e-2):
    P.eps=eps; P.s0=P.s_eps(P.x0)
    X,U,l,d=P.run_dmpc(m,True); R['eps'][str(eps)]=dict(max_state_diff=float(np.max(np.abs(X-Xe))),delay=float(1/X[-1,0]),minacc=float(P.access(X[-1]).min()))
    print('eps',eps,'done',time.time()-t0); sys.stdout.flush()
P.eps=1e-3; P.s0=P.s_eps(P.x0)
R['tgo']={}
for tau,gam,om in ((40.0,20.0,0.5),(20.0,10.0,0.5),(20.0,20.0,0.8)):
    P.tau,P.gamma,P.omega=tau,gam,om
    X,U,l,d=P.run_dmpc(m,True)
    R['tgo'][f'{tau}_{gam}_{om}']=dict(maxU_diff=float(np.max(np.abs(U-Ue))),rel_cum=float(np.abs(U.sum(0)-Ue.sum(0)).sum()/Ue.sum()),delay=float(1/X[-1,0]),maxit=int(d['max_it']),meanit=float(d['mean_it']),nonconv=int(d['nonconv']))
    print('tgo',tau,gam,om,'done',time.time()-t0); sys.stdout.flush()
P.tau,P.gamma,P.omega=20.0,20.0,0.5
R['init']={}
base=Ue.sum(axis=0)
rng=np.random.default_rng(2026090601)
inits={'zeros':(np.zeros((6,4)),np.zeros(4)),'uniform_0.03':(np.full((6,4),0.03),np.zeros(4)),'sq_lambda2':(np.tile(P.uSQ[:,None],(1,4)),np.full(4,2.0)),'random_seed_2026090601':(rng.uniform(0,0.03,(6,4)),np.zeros(4))}
for lab,(U0,l0) in inits.items():
    X,U,l,d=P.run_dmpc(m,True,U0=U0,lam0=l0)
    R['init'][lab]=dict(rel_l1=float(np.abs(U.sum(0)-base).sum()/base.sum()),delay=float(1/X[-1,0]),minacc=float(P.access(X[-1]).min()),maxit=int(d['max_it']),nonconv=int(d['nonconv']),max_state_diff=float(np.max(np.abs(X-Xe))))
    print('init',lab,'done',time.time()-t0); sys.stdout.flush()
R['autpert']={}
for H in (100,300):
    rat=[]
    for j in range(8):
        for sgn in (1,-1):
            x=P.x0.copy(); x[j]+=sgn*0.01
            for t in range(H): x=m.step(x,np.zeros(6))
            rat.append(float(np.linalg.norm(x-P.x0)/0.01))
    R['autpert'][str(H)]=dict(min=min(rat),max=max(rat))
X,U,l,d=P.run_dmpc(m,False,T_=200); R['long_plain']=dict(x200=[float(v) for v in X[-1]],delay200=float(1/X[-1,0]),u200=[float(v) for v in U[-1]],x36=[float(v) for v in X[36]])
print('long done',time.time()-t0); sys.stdout.flush()
json.dump(R,open('v16_results_C.json','w'),indent=1)

perm=np.array([3,5,0,4,1,2])
Q0,r0,Bu0,uSQ0=P.Q.copy(),P.r.copy(),P.Bu.copy(),P.uSQ.copy()
P.Q=Q0[perm]; P.r=r0[perm]; P.uSQ=uSQ0[perm]
mp=P.Model(Bu=Bu0[:,perm],Q=P.Q,r=P.r)
Xp,Up,lp,dp=P.run_dmpc(mp,True)
Uback=np.zeros_like(Up); Uback[:,perm]=Up
print("label permutation: max|U-Ue| =",np.max(np.abs(Uback-Ue))," max|X-Xe| =",np.max(np.abs(Xp-Xe)))
P.Q,P.r,P.uSQ=Q0,r0,uSQ0
