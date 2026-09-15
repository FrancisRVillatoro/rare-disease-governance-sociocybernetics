"""Regenerate the manuscript/Supplement figures from the canonical v16 outputs.
Run pilot16.py and audit16.py first; run montecarlo16.py + summarize16.py before Fig. S6.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pilot16 as P

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

def load(name): return np.load(HERE / name)
def acc_series(X): return np.array([P.access(x) for x in X])

Xs,Xd,Xe = load('v16_Xs.npy'),load('v16_Xd.npy'),load('v16_Xe.npy')
Ud,Ue,le = load('v16_Ud.npy'),load('v16_Ue.npy'),load('v16_le.npy')
Xsqs,Xce,Xcf,Xc = load('v16_Xsqs.npy'),load('v16_Xce.npy'),load('v16_Xcf.npy'),load('v16_Xc.npy')
t=np.arange(len(Xs))

fig,ax=plt.subplots(figsize=(8.2,5.0))
for X,label in [(Xs,'Status quo'),(Xsqs,'Scaled status quo (equal mean spend)'),(Xce,'Best constant policy (equal mean spend)'),(Xcf,'Best constant policy (full envelope)'),(Xc,'Centralised MPC'),(Xd,'Symmetric-price DMPC'),(Xe,'Access-screen DMPC')]:
    ax.plot(t,1/X[:,0],label=label)
ax.axhline(1.0,linestyle='--',linewidth=1.0,label='One-year external reference')
ax.set_xlabel('Decision period'); ax.set_ylabel('Model-derived diagnostic-delay proxy (years)'); ax.set_ylim(bottom=0)
ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(FIG/'fig1_delay_proxy.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,5.0))
labels=['x_M medical','x_S scientific','x_P policy','x_L legal','x_E evidence/access','x_V visibility','x_C civil/registry','x_O organisational']
for j,label in enumerate(labels): ax.plot(t,Xe[:,j],label=label)
ax.set_xlabel('Decision period'); ax.set_ylabel('Normalised state'); ax.set_ylim(0,1); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(FIG/'figS1_states_screen.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,4.8))
for X,label in [(Xs,'Status quo'),(Xd,'Symmetric-price DMPC'),(Xe,'Access-screen DMPC')]: ax.plot(t,acc_series(X).min(1),label=label)
ax.axhline(P.abar,linestyle='--',linewidth=1.0,label='Illustrative access floor 0.40')
ax.set_xlabel('Decision period'); ax.set_ylabel('Minimum access score'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'figS2_min_access.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,4.8))
for X,label in [(Xs,'Status quo'),(Xd,'Symmetric-price DMPC'),(Xe,'Access-screen DMPC')]:
    A=acc_series(X); ax.plot(t,np.ptp(A,axis=1),label=label)
ax.axhline(0.12,linestyle='--',linewidth=1.0,label='Conceptual dispersion ceiling 0.12 (inactive)')
ax.set_xlabel('Decision period'); ax.set_ylabel('Access-score dispersion'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'figS3_dispersion.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,4.8)); tt=np.arange(len(Ue)); names=['u_H health services','u_R regulatory','u_HTA HTA/access','u_B budgetary','u_V visibility/participation','u_O organisational']
for j,label in enumerate(names): ax.plot(tt,Ue[:,j],label=label)
ax.set_xlabel('Decision period'); ax.set_ylabel('Applied intervention intensity'); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(FIG/'figS4_interventions_screen.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,4.5)); ax.plot(tt,le,marker='o',markersize=3); ax.axhline(0,linewidth=.8)
ax.set_xlabel('Decision period'); ax.set_ylabel('Fiscal shadow price, first horizon step'); fig.tight_layout(); fig.savefig(FIG/'figS5_shadow_price.png',dpi=300); plt.close(fig)

mc_path=HERE/'v16_mc.json'
if mc_path.exists():
    rows=json.loads(mc_path.read_text())
    if rows:
        fig,axs=plt.subplots(1,2,figsize=(9.2,4.5))
        axs[0].boxplot([[r['sq_delay'] for r in rows],[r['d_delay'] for r in rows],[r['e_delay'] for r in rows]],tick_labels=['Status quo','DMPC','Access screen'])
        axs[0].set_ylabel('Final delay proxy (years)')
        axs[1].boxplot([[r['sq_min'] for r in rows],[r['d_min'] for r in rows],[r['e_min'] for r in rows]],tick_labels=['Status quo','DMPC','Access screen'])
        axs[1].axhline(P.abar,linestyle='--',linewidth=1.0); axs[1].set_ylabel('Final minimum access')
        fig.tight_layout(); fig.savefig(FIG/'figS6_montecarlo.png',dpi=300); plt.close(fig)
print('Figures written to',FIG)
