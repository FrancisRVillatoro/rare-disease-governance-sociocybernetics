"""Independent vectorised objective and forward-sensitivity gradient for BR audits.

The physical model, weights, bounds and predictions equal release_source/pilot16.py.
There is NO proximal term here. Objective equality and derivatives are checked
numerically against the unmodified reference implementation by the audit driver.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'release_source'))
import pilot16 as P

ACCESS_W=np.array([[.42,0,0,.18,.18,0,.10,.12],
                   [.38,0,0,.16,.16,.10,.16,.14],
                   [.40,.16,0,.20,0,0,.10,.14]])
ACCESS_OFFSET=np.array([0.,-.10,0.])

class LocalObjective:
    def __init__(self,i,U,x,screen,lam=None):
        self.i=int(i);self.U=np.array(U,dtype=float);self.x=np.array(x,dtype=float)
        self.screen=bool(screen);self.lam=np.zeros(P.Np) if lam is None else np.asarray(lam)
        self.m=P.Model()
        self.Bbase=P.Bu@self.U
        self.Q=P.Q[self.i]
    def reference_value(self,ui):
        return float(P.J_local(np.asarray(ui),self.i,self.U,self.x,self.m,self.screen)+self.lam@ui)
    def value_grad(self,ui):
        ui=np.asarray(ui);xx=self.x.copy();S=np.zeros((8,P.Np));cost=0.;grad=np.zeros(P.Np)
        for k in range(P.Np):
            z=xx-P.x0;az=P.Ac@z;kap=P.cO*(.80+.40*xx[7])
            g=np.tanh(P.ks*(xx-P.theta));root=np.sqrt(g*g+P.eps*P.eps)
            act=.5*(g+root);dact=.5*(1+g/root)*P.ks*(1-g*g)
            raw=P.x0+z-P.Lam*z+kap*az+self.Bbase[:,k]+P.Bu[:,self.i]*(ui[k]-self.U[self.i,k])+P.delta*P.Fd*(act-P.s0)
            A=np.diag(1-P.Lam+P.delta*P.Fd*dact)+kap*P.Ac
            A[:,7]+=.4*P.cO*az
            S=A@S;S[:,k]+=P.Bu[:,self.i]
            S*=((raw>0.)&(raw<1.))[:,None]
            xx=np.clip(raw,0.,1.);e=xx-P.xstar
            cost+=np.sum(self.Q*e*e)+P.r[self.i]*ui[k]**2
            grad+=2*(self.Q*e)@S;grad[k]+=2*P.r[self.i]*ui[k]
            if self.screen:
                a=ACCESS_W@xx+ACCESS_OFFSET;W=ACCESS_W.copy()
                if a[1]<=0. or a[1]>=1.: W[1,:]=0.
                a[1]=np.clip(a[1],0.,1.)
                short=np.maximum(P.abar-a,0.)
                cost+=P.rho_a*np.sum(short*short)
                grad-=2*P.rho_a*(short@W)@S
        return float(cost+self.lam@ui),grad+self.lam
    def batch(self,ui):
        ui=np.asarray(ui,dtype=float);scalar=ui.ndim==1
        if scalar:ui=ui[:,None]
        xx=np.broadcast_to(self.x[:,None],(8,ui.shape[1])).copy();cost=np.zeros(ui.shape[1])
        for k in range(P.Np):
            z=xx-P.x0[:,None];g=np.tanh(P.ks*(xx-P.theta[:,None]));act=.5*(g+np.sqrt(g*g+P.eps*P.eps))
            raw=P.x0[:,None]+z-P.Lam[:,None]*z+(P.cO*(.8+.4*xx[7]))[None,:]*(P.Ac@z)+self.Bbase[:,k,None]+P.Bu[:,self.i,None]*(ui[k]-self.U[self.i,k])[None,:]+P.delta*P.Fd[:,None]*(act-P.s0[:,None])
            xx=np.clip(raw,0.,1.);e=xx-P.xstar[:,None]
            cost+=np.sum(self.Q[:,None]*e*e,axis=0)+P.r[self.i]*ui[k]**2
            if self.screen:
                a=ACCESS_W@xx+ACCESS_OFFSET[:,None];a[1]=np.clip(a[1],0.,1.)
                cost+=P.rho_a*np.sum(np.maximum(P.abar-a,0.)**2,axis=0)
        cost+=self.lam@ui
        return float(cost[0]) if scalar else cost
    def predicted_states(self,ui):
        U=self.U.copy();U[self.i]=ui;xx=self.x.copy();states=[]
        for k in range(P.Np):
            xx=self.m.step(xx,U[:,k]);states.append(xx.copy())
        return np.array(states)
