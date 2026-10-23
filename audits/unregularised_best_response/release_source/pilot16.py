import numpy as np, time, sys
from scipy.optimize import minimize

# ---------------- specification v16 (Supplementary Material) ----------------
x0 = np.array([0.213, 0.115, 0.405, 0.698, 0.540, 0.193, 0.172, 0.646])
xstar = np.array([0.550, 0.360, 0.550, 0.740, 0.620, 0.240, 0.240, 0.720])
Lam = np.array([0.055, 0.050, 0.050, 0.040, 0.045, 0.050, 0.050, 0.040])
Fd = np.array([0.30, 0.26, 0.24, 0.20, 0.22, 0.20, 0.20, 0.28])
theta = np.array([0.30, 0.25, 0.35, 0.55, 0.45, 0.25, 0.25, 0.55])
Ac = np.array([
 [0.00,0.30,0.16,0.00,0.00,0.00,0.00,0.24],
 [0.10,0.00,0.12,0.00,0.00,0.00,0.00,0.16],
 [0.00,0.00,0.00,0.00,0.10,0.28,0.22,0.12],
 [0.00,0.12,0.18,0.00,0.00,0.00,0.00,0.10],
 [0.00,0.10,0.12,0.28,0.00,0.00,0.00,0.12],
 [0.00,0.18,0.12,0.00,0.00,0.00,0.18,0.10],
 [0.10,0.00,0.00,0.00,0.00,0.20,0.00,0.18],
 [0.08,0.00,0.08,0.00,0.00,0.00,0.10,0.00]])
Bu = np.array([
 [0.204,0,0,0,0,0],
 [0,0,0,0.060,0,0],
 [0,0,0,0.144,0,0],
 [0,0.180,0,0,0,0],
 [0,0,0.180,0,0,0],
 [0,0,0,0,0.204,0],
 [0,0,0,0,0.180,0],
 [0.060,0,0,0,0,0.204]])
Q = np.array([
 [5.00,0.15,0.80,0.10,0.10,0.05,0.05,1.20],
 [0.15,0.15,0.80,5.00,0.80,0.05,0.05,0.20],
 [0.20,0.40,0.30,1.00,5.00,0.05,0.05,0.20],
 [0.80,0.30,4.00,0.50,1.20,0.10,0.10,0.50],
 [0.10,0.20,0.50,0.10,0.10,4.00,4.00,0.20],
 [0.80,0.20,0.50,0.20,0.20,0.20,0.80,5.00]])
r = np.full(6, 14.0)
uSQ = np.array([0.010,0.006,0.006,0.008,0.006,0.008])
cO, delta, ks, eps = 0.068, 0.004, 4.0, 1e-3
umax, usum = 0.08, 0.18
abar, rho_a = 0.40, 400.0
T, Np, Na = 36, 4, 6
tau, gamma, omega = 20.0, 20.0, 0.5
tol_p, tol_U, tol_l, max_it = 1e-6, 1e-5, 1e-5, 250
slsqp_opts = {'maxiter':150, 'ftol':1e-10}

def kappaO(xO): return cO*(0.80+0.40*xO)
def s_eps(x):
    g = np.tanh(ks*(x-theta)); return 0.5*(g+np.sqrt(g*g+eps*eps))
s0 = s_eps(x0)

class Model:
    def __init__(self, Lam=Lam, Bu=Bu, xstar=xstar, r=r, Q=Q, Np=Np):
        self.Lam, self.Bu, self.xstar, self.r, self.Q, self.Np = Lam, Bu, xstar, r, Q, Np
    def step(self, x, u, w=None):
        z = x - x0
        zt = z - self.Lam*z + kappaO(x[7])*(Ac@z) + self.Bu@u + delta*Fd*(s_eps(x)-s0)
        if w is not None: zt = zt + w
        return np.clip(x0+zt, 0.0, 1.0)

def access(x):
    aHV = 0.42*x[0]+0.18*x[3]+0.18*x[4]+0.10*x[6]+0.12*x[7]
    aLV = np.clip(0.38*x[0]+0.16*x[3]+0.16*x[4]+0.16*x[6]+0.14*x[7]-0.10*(1-x[5]),0,1)
    aPG = 0.40*x[0]+0.16*x[1]+0.20*x[3]+0.10*x[6]+0.14*x[7]
    return np.array([aHV,aLV,aPG])

def J_local(ui, i, U, x, m, screen):
    Uc = U.copy(); Uc[i,:] = ui
    xx = x.copy(); J = 0.0
    for k in range(m.Np):
        xx = m.step(xx, Uc[:,k]); e = xx - m.xstar
        J += np.sum(m.Q[i]*e*e) + m.r[i]*ui[k]**2
        if screen: J += rho_a*np.sum(np.maximum(0.0, abar-access(xx))**2)
    return J

def coordinate(x, U, lam, m, screen):
    """Proximal Jacobi primal-dual coordination for one decision period. Returns U, lam, diagnostics."""
    Np_ = m.Np
    for q in range(1, max_it+1):
        Uold, lamold = U.copy(), lam.copy()
        Uhalf = np.zeros_like(U)
        for i in range(Na):
            obj = lambda ui: J_local(ui, i, Uold, x, m, screen) + lam@ui + 0.5*tau*np.sum((ui-Uold[i])**2)
            res = minimize(obj, Uold[i], method='SLSQP', bounds=[(0.0, umax)]*Np_, options=slsqp_opts)
            Uhalf[i] = np.clip(res.x, 0.0, umax)
        U = (1-omega)*Uold + omega*Uhalf
        lam = np.maximum(0.0, lam + gamma*(U.sum(axis=0) - usum))
        rp = np.max(np.maximum(0.0, U.sum(axis=0)-usum)); rU = np.max(np.abs(U-Uold)); rl = np.max(np.abs(lam-lamold))
        if rp <= tol_p and rU <= tol_U and rl <= tol_l:
            return U, lam, dict(it=q, conv=True, rp=rp, rU=rU, rl=rl)
    return U, lam, dict(it=max_it, conv=False, rp=rp, rU=rU, rl=rl)

def run_dmpc(m, screen, U0=None, lam0=None, lam_warm=True, T_=T, verbose=False):
    x = x0.copy(); X=[x.copy()]; Uapp=[]; lams=[]; its=[]; nonconv=0; maxproj=0.0; maxrp=maxrU=maxrl=0.0
    U = np.tile(uSQ[:,None],(1,m.Np)) if U0 is None else U0.copy()
    lam = np.zeros(m.Np) if lam0 is None else lam0.copy()
    for t in range(T_):
        if t>0:
            U = np.concatenate([U[:,1:],U[:,-1:]],axis=1)
            lam = np.concatenate([lam[1:],lam[-1:]]) if lam_warm else np.zeros(m.Np)
        U, lam, d = coordinate(x, U, lam, m, screen)
        its.append(d['it']); nonconv += (not d['conv']); maxrp=max(maxrp,d['rp']); maxrU=max(maxrU,d['rU']); maxrl=max(maxrl,d['rl'])
        u = U[:,0].copy()
        if u.sum() > usum:
            viol = u.sum()-usum
            if viol <= 2e-6:
                maxproj = max(maxproj, viol); u = u*usum/u.sum(); U[:,0] = u
            else:
                nonconv += 1
        Uapp.append(u); lams.append(lam[0])
        x = m.step(x, u); X.append(x.copy())
    return np.array(X), np.array(Uapp), np.array(lams), dict(mean_it=np.mean(its), max_it=np.max(its), nonconv=nonconv, maxrp=maxrp, maxrU=maxrU, maxrl=maxrl, maxproj=maxproj)

def run_const(m, u, T_=T):
    x = x0.copy(); X=[x.copy()]
    for t in range(T_): x = m.step(x,u); X.append(x.copy())
    return np.array(X), np.tile(u,(T_,1))

def realised_cost(m, X, Uapp, screen=False):
    J = 0.0
    for t in range(len(Uapp)):
        e = X[t+1]-m.xstar
        J += sum(np.sum(m.Q[i]*e*e) + m.r[i]*Uapp[t,i]**2 for i in range(Na))
        if screen: J += Na*rho_a*np.sum(np.maximum(0.0, abar-access(X[t+1]))**2)
    return J

def summary(name, m, X, Uapp, extra=''):
    a = np.array([access(x) for x in X]); amin=a.min(axis=1); phi=a.max(axis=1)-a.min(axis=1)
    first = next((t for t in range(len(X)) if amin[t]>=0.40), None); tot=Uapp.sum(axis=1)
    print(f"{name:34s} delay={1/X[-1,0]:.4f} minacc={amin[-1]:.4f} Phi={phi[-1]:.4f} meanU={tot.mean():.4f} maxU={tot.max():.4f} first>=0.40={first} J={realised_cost(m,X,Uapp):.2f} {extra}")

if __name__ == '__main__':
    m = Model()
    def f_aut(x): return m.step(x, np.zeros(6))
    ep=1e-7; J=np.zeros((8,8))
    for j in range(8):
        e=np.zeros(8); e[j]=ep; J[:,j]=(f_aut(x0+e)-f_aut(x0-e))/(2*ep)
    print("rho(J_f(x0)) =", max(abs(np.linalg.eigvals(J))))
    Xs,Us = run_const(m, uSQ); summary('status quo', m, Xs, Us)
    for lw in (True, False):
        t0=time.time(); Xd,Ud,ld,dd = run_dmpc(m, False, lam_warm=lw); summary(f'symmetric-price DMPC (lam_warm={lw})', m, Xd, Ud, f"{dd} {time.time()-t0:.0f}s")
        t0=time.time(); Xe,Ue,le,de = run_dmpc(m, True, lam_warm=lw); summary(f'access-screen DMPC (lam_warm={lw})', m, Xe, Ue, f"{de} {time.time()-t0:.0f}s")
        print("   screen: lambda_0(t) first 10:", np.round(le[:10],4), " max", le.max(), " positive periods:", np.where(le>1e-9)[0])
        print("   cumulative allocation DMPC  :", np.round(Ud.sum(axis=0),4))
        print("   cumulative allocation screen:", np.round(Ue.sum(axis=0),4))
        np.save(f'Xd_{int(lw)}.npy',Xd); np.save(f'Ud_{int(lw)}.npy',Ud); np.save(f'Xe_{int(lw)}.npy',Xe); np.save(f'Ue_{int(lw)}.npy',Ue)
