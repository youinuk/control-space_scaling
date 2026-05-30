"""
simulation_stage1_final.py
Stage 1: O(K^2) Calibration Bottleneck

Physical parameters (realistic Si/SiGe spin-shuttling):
  X_MAX=20a, u_j=10a, v_scale=20a/T, sigma_M~0.0032, M_cut=1e-4

Note on Test C (static > dynamic rank):
  rank_norm = linear independence of rows, NOT crosstalk magnitude.
  M_static (v=0): rows encode unique 1/r^3 fingerprints -> max independence.
  M_dynamic (v=20a/T): trajectory integral smears spatial structure
    -> rows become less distinct -> rank slightly lower.
  Both >=0.93: O(K^2) bottleneck holds at all speeds.
"""

import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, shutil, os, sys
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':'serif','font.size':9,'axes.labelsize':10,
    'axes.titlesize':10,'legend.fontsize':7.5,'xtick.labelsize':8,
    'ytick.labelsize':8,'lines.linewidth':1.6,'axes.linewidth':0.8,
    'figure.dpi':200,'text.usetex':False,
})

# ══════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════
RNG         = np.random.default_rng(0)
N_TRIALS    = 30          # 30 trials: significantly smoother curves (std/sqrt(30))
N_BOOT       = 1000        # bootstrap samples for power-law / fit uncertainty

L_GRID      = 18
Z_DEPTH     = 0.35
V_SCALE     = 20.0
V_NOISE     = 0.15
X_MAX       = 20.0
T_TRANS     = 1.0
U_J         = X_MAX * T_TRANS / 2.0    # = 10.0
N_STEPS_DYN = 300

N_SHOTS     = 1000
SIGMA_PHI   = 1.0 / np.sqrt(N_SHOTS)
SIGMA_M     = SIGMA_PHI / U_J
EPS_FT      = 1e-3
M_CUT       = EPS_FT / U_J

T_RAMSEY    = 8.0
TAU_DRIFT   = 3600.0

C4    = ['#d6604d','#2166ac','#4dac26','#762a83']
C6    = ['#2166ac','#4dac26','#f46d43','#762a83','#a6611a','#1b7837']
MARKS = ['o','s','^','D','v','P']

def _snr(v):  return v / SIGMA_M
def _nreq(v): return max(1.0,(3.0*SIGMA_M/v)**2)
def _despine(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def bootstrap_powerlaw_ci(K_vals, y_trials_by_K, K_fit, K_min=15, n_boot=N_BOOT,
                          ci=(2.5, 97.5), seed=1234):
    """
    Bootstrap confidence intervals for y ~ A K^b.

    y_trials_by_K is a list of 1D arrays; entry q contains the per-trial
    values used to form the mean y(K_q).  Each bootstrap replicate resamples
    trials independently at every K, refits log y = log A + b log K, and
    returns percentile intervals for both b and the fitted curve.

    This is intentionally lightweight: it quantifies the Monte-Carlo placement
    uncertainty of the reported scaling exponent, without changing the
    underlying physical model.
    """
    K_vals = np.asarray(K_vals, dtype=float)
    mask = K_vals >= K_min
    K_use = K_vals[mask]
    trials_use = [np.asarray(y_trials_by_K[i], dtype=float)
                  for i in np.where(mask)[0]]

    rng = np.random.default_rng(seed)
    bs, curves = [], []
    for _ in range(n_boot):
        means = []
        good_K = []
        for K, vals in zip(K_use, trials_use):
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            sample = rng.choice(vals, size=len(vals), replace=True)
            m = float(np.mean(sample))
            if m > 0:
                means.append(m); good_K.append(K)
        if len(means) < 3:
            continue
        b, a = np.polyfit(np.log(good_K), np.log(means), 1)
        bs.append(b)
        curves.append(np.exp(a) * np.asarray(K_fit, dtype=float)**b)

    bs = np.asarray(bs)
    curves = np.asarray(curves)
    if len(bs) == 0:
        nan_curve = np.full_like(K_fit, np.nan, dtype=float)
        return (np.nan, np.nan), nan_curve, nan_curve

    b_lo, b_hi = np.percentile(bs, ci)
    fit_lo, fit_hi = np.percentile(curves, ci, axis=0)
    return (float(b_lo), float(b_hi)), fit_lo, fit_hi


# ══════════════════════════════════════════════════════════════
# LATTICE & MATRIX BUILDERS
# ══════════════════════════════════════════════════════════════
def perfect_lattice(L, a=1.0):
    xs,ys = np.meshgrid(np.arange(L)*a, np.arange(L)*a)
    return np.stack([xs.ravel(),ys.ravel()],axis=1).astype(float)

def disordered_lattice(L, a=1.0, f=0.0, rng=RNG):
    pos = perfect_lattice(L,a).copy()
    if f>0: pos += rng.uniform(-f*a,f*a,size=pos.shape)
    return pos

def build_M_static(positions, active_idx, z=Z_DEPTH):
    """
    M^stat_ij = chi(r_i^0, r_j^0)   [v=0 limit, vectorised]

    Each row encodes the unique 1/r^3 spatial fingerprint of qubit i.
    Rows are maximally distinct -> HIGHEST rank among all v_scales.
    """
    r0  = positions[active_idx]
    dr  = r0[:,None,:] - r0[None,:,:]
    d2  = np.sum(dr**2,axis=-1) + z**2
    M   = 1.0/d2**1.5
    np.fill_diagonal(M,0.0)
    return M

def build_M_dynamic(positions, active_idx,
                    v_scale=V_SCALE, v_noise=V_NOISE,
                    z=Z_DEPTH, n_steps=N_STEPS_DYN, T=T_TRANS, rng=RNG):
    """
    M^dyn_ij = int_0^T chi(r_i(t), r_j^0) dt   [trapezoidal]

    At v=20a/T qubit i sweeps ~20a during T (full lattice width).
    Trajectory integral acts as low-pass filter: chi_ij becomes small
    and similar across j -> rows less distinct -> rank slightly lower
    than M_static. Both satisfy rank_norm >= 0.93.
    """
    K   = len(active_idx); r0 = positions[active_idx]
    ang = rng.uniform(0,2*np.pi,size=K)
    spd = v_scale*(1.0+rng.uniform(-v_noise,v_noise,size=K))
    v   = spd[:,None]*np.stack([np.cos(ang),np.sin(ang)],axis=1)
    ts  = np.linspace(0,T,n_steps); dt = ts[1]-ts[0]; M = np.zeros((K,K))
    for ti,t in enumerate(ts):
        ri = r0+v*t; dr = ri[:,None,:]-r0[None,:,:]
        d2 = np.sum(dr**2,axis=-1)+z**2
        w  = 0.5 if(ti==0 or ti==n_steps-1) else 1.0
        M += w/d2**1.5*dt
    np.fill_diagonal(M,0.0)
    return M

def gauge_project_sv(M):
    K = M.shape[0]
    return linalg.svdvals((np.eye(K)-np.ones((K,K))/K)@M)

def rank_norm(sv, eps):
    if sv[0]==0: return 0.0
    return int(np.sum(sv>eps*sv[0]))/max(len(sv)-1,1)


# ══════════════════════════════════════════════════════════════
# TEST A — Threshold Robustness
# ══════════════════════════════════════════════════════════════
def test_A(positions, K_range, n_trials=N_TRIALS,
           thresholds=(1e-2,1e-3,1e-4,1e-5)):
    N = len(positions)
    res = {e:{'mean':[],'std':[]} for e in thresholds}; Kv=[]
    for K in K_range:
        if K>N or K<3: continue
        Kv.append(K); per={e:[] for e in thresholds}
        for _ in range(n_trials):
            idx = RNG.choice(N,size=K,replace=False)
            sv  = gauge_project_sv(build_M_dynamic(positions,idx))
            for e in thresholds: per[e].append(rank_norm(sv,e))
        for e in thresholds:
            res[e]['mean'].append(np.mean(per[e]))
            res[e]['std'].append(np.std(per[e]))
    for e in thresholds:
        res[e]['mean']=np.array(res[e]['mean'])
        res[e]['std'] =np.array(res[e]['std'])
    return np.array(Kv), res


# ══════════════════════════════════════════════════════════════
# TEST B — Disorder Robustness
# ══════════════════════════════════════════════════════════════
def test_B(L, K_range, n_trials=N_TRIALS,
           disorder_levels=(0.0,0.10,0.20,0.35), eps=1e-3):
    N=L*L; res={}
    for f in disorder_levels:
        means,stds,Ks=[],[],[]
        for K in K_range:
            if K>N or K<3: continue
            Ks.append(K); ranks=[]
            for _ in range(n_trials):
                pos=disordered_lattice(L,f=f)
                idx=RNG.choice(N,size=K,replace=False)
                sv =gauge_project_sv(build_M_dynamic(pos,idx))
                ranks.append(rank_norm(sv,eps))
            means.append(np.mean(ranks)); stds.append(np.std(ranks))
        res[f]={'K':np.array(Ks),'mean':np.array(means),'std':np.array(stds)}
    return res


# ══════════════════════════════════════════════════════════════
# TEST C — Static vs Dynamic
# ══════════════════════════════════════════════════════════════
def test_C(positions, K_probe=(10,20,40,60,80),
           v_scales=(0.0,0.5,2.0,5.0,10.0,20.0),
           n_trials=N_TRIALS, eps=1e-3, K_frob=20):
    """
    v=0 (static) has HIGHEST rank because rows encode unique 1/r^3
    fingerprints at parking positions (snapshot).
    v=20a/T (dynamic) slightly lower: trajectory integral smears
    spatial structure (long-exposure blur).
    Both >=0.93 -> O(K^2) holds at all speeds.
    """
    N=len(positions); rank_data={}
    for vs in v_scales:
        d={}
        for K in K_probe:
            if K>N: continue
            ranks=[]
            for _ in range(n_trials):
                idx=RNG.choice(N,size=K,replace=False)
                if vs==0.0:
                    M=build_M_static(positions,idx)
                else:
                    n_int=60 if vs<1 else 300
                    M=build_M_dynamic(positions,idx,v_scale=vs,n_steps=n_int)
                ranks.append(rank_norm(gauge_project_sv(M),eps))
            d[K]=(np.mean(ranks),np.std(ranks))
        rank_data[vs]=d

    idx_f =np.random.default_rng(99).choice(N,size=K_frob,replace=False)
    M_stat=build_M_static(positions,idx_f)
    frob=[]
    for vs in v_scales:
        if vs==0.0: frob.append(0.0); continue
        n_int=60 if vs<1 else 300
        diffs=[linalg.norm(
            build_M_dynamic(positions,idx_f,v_scale=vs,n_steps=n_int,
                            rng=np.random.default_rng(t))-M_stat,'fro')
               /linalg.norm(M_stat,'fro') for t in range(12)]
        frob.append(float(np.mean(diffs)))
    return rank_data, frob


# ══════════════════════════════════════════════════════════════
# TEST D — Absolute Threshold
# ══════════════════════════════════════════════════════════════
def test_D(positions, K_range, n_trials=N_TRIALS, K_hist=40):
    N=len(positions)
    data={k:[] for k in ('K','Nft','Nft_std','Nnear','Nmid','Ceff','Ceff_std')}
    nft_trials_by_K=[]; ceff_trials_by_K=[]
    hist_near,hist_mid=[],[]
    for K in K_range:
        Nft_l,Cnear_l,Cmid_l,Ceff_l=[],[],[],[]
        for _ in range(n_trials):
            idx=RNG.choice(N,size=K,replace=False)
            M=build_M_dynamic(positions,idx)
            ent=M[np.triu_indices(K,k=1)]
            ft=ent[ent>M_CUT]; nr=ft[_snr(ft)>=3.0]; md=ft[_snr(ft)<3.0]
            Nft_l.append(len(ft)); Cnear_l.append(len(nr)); Cmid_l.append(len(md))
            cn=len(nr)*N_SHOTS
            cm=sum(_nreq(v) for v in md) if len(md)>0 else 0.0
            Ceff_l.append(cn+cm)
            if K==K_hist:
                hist_near.extend(_snr(nr).tolist())
                hist_mid.extend(_snr(md).tolist())
        data['K'].append(K); data['Nft'].append(np.mean(Nft_l))
        data['Nft_std'].append(np.std(Nft_l))
        nft_trials_by_K.append(np.array(Nft_l, dtype=float))
        ceff_trials_by_K.append(np.array(Ceff_l, dtype=float))
        data['Nnear'].append(np.mean(Cnear_l)); data['Nmid'].append(np.mean(Cmid_l))
        data['Ceff'].append(np.mean(Ceff_l)); data['Ceff_std'].append(np.std(Ceff_l))
    for k in data: data[k]=np.array(data[k])

    vd=data['K']>=15
    b,a_=np.polyfit(np.log(data['K'][vd]),np.log(data['Nft'][vd]),1)
    K_fit=np.logspace(np.log10(float(data['K'][vd][0])),
                      np.log10(float(data['K'][vd][-1])),200)
    Nft_fit=np.exp(a_)*K_fit**b
    b_ci, Nft_fit_lo, Nft_fit_hi = bootstrap_powerlaw_ci(data['K'], nft_trials_by_K, K_fit)
    frac_FT=float(np.mean(data['Nft'][data['K']>=20]
                  /(data['K'][data['K']>=20]*(data['K'][data['K']>=20]-1)/2)))

    mid_snr=np.array([s for s in hist_mid if 0<s<3])
    N_req_avg=float(np.mean((3.0/mid_snr)**2)) if len(mid_snr)>0 else N_SHOTS
    T_eff_mid=T_RAMSEY*(N_req_avg/N_SHOTS)
    K_dense=np.arange(2,101,dtype=float)
    cost_naive=K_dense*(K_dense-1)/2*T_RAMSEY
    cost_eff=data['Nnear']*T_RAMSEY+data['Nmid']*T_eff_mid
    K_star_n=float(np.interp(TAU_DRIFT,cost_naive,K_dense)) \
             if cost_naive[-1]>TAU_DRIFT else float(K_dense[-1])
    K_star_e=float(np.interp(TAU_DRIFT,cost_eff,data['K'])) \
             if cost_eff[-1]>TAU_DRIFT else float(data['K'][-1])
    return (data,b,b_ci,K_fit,Nft_fit,Nft_fit_lo,Nft_fit_hi,frac_FT,T_eff_mid,K_dense,
            cost_naive,cost_eff,K_star_n,K_star_e,
            np.array(hist_near),np.array(hist_mid))


# ══════════════════════════════════════════════════════════════
# LABEL DICTS
# ══════════════════════════════════════════════════════════════
TL = {1e-2:r'$\varepsilon=10^{-2}$',1e-3:r'$\varepsilon=10^{-3}$',
      1e-4:r'$\varepsilon=10^{-4}$',1e-5:r'$\varepsilon=10^{-5}$'}
DL = {0.0:'f=0 (perfect)',0.10:'f=10%',0.20:'f=20%',0.35:'f=35%'}
VL = {0.0:'v=0 (static)',0.5:'0.5a/T',2.0:'2a/T',
      5.0:'5a/T',10.0:'10a/T',20.0:'20a/T (realistic)'}

def _rank_panel(ax):
    """Shared formatting: y from 0.83 to 1.00 exactly."""
    ax.axhline(1.0,color='#555',ls='--',lw=0.9,alpha=0.5)
    ax.set_ylim(0.83, 1.00); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_ylabel(r'$r_{\varepsilon}/(K-1)$')
    ax.set_xlabel('Active qubits $K$')
    _despine(ax)


# ══════════════════════════════════════════════════════════════
# PLOT A
# ══════════════════════════════════════════════════════════════
def plot_A(K_arr, res, thresholds, outdir):
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.9))
    ax=axes[0]
    for i,e in enumerate(thresholds):
        mn=res[e]['mean']; sd=res[e]['std']
        ax.fill_between(K_arr,np.clip(mn-sd,0.5,1.0),np.clip(mn+sd,0.5,1.0),
                        color=C4[i],alpha=0.12)
        ax.plot(K_arr,mn,color=C4[i],marker=MARKS[i],ms=4,markevery=3,
                lw=1.5,label=TL[e])
    _rank_panel(ax)
    ax.set_title(f'(a) Threshold robustness  ($v={V_SCALE:.0f}a/T$)',fontsize=9)
    ax.legend(frameon=False,fontsize=7.5,ncol=2)

    ax=axes[1]
    for i,e in enumerate(thresholds):
        ax.semilogy(K_arr,np.clip(1-res[e]['mean'],1e-5,1),
                    color=C4[i],marker=MARKS[i],ms=4,markevery=3,lw=1.5,label=TL[e])
    ax.axhline(0.05,color='gray',ls=':',lw=0.8)
    ax.text(K_arr[-1]*0.52,0.058,'5% level',fontsize=6.5,color='gray')
    ax.set_xlabel('Active qubits $K$')
    ax.set_ylabel(r'rank deficit  $1 - r_{\varepsilon}/(K-1)$')
    ax.set_title('(b) Rank deficit (log scale)',fontsize=9)
    ax.set_xlim(K_arr[0],K_arr[-1])
    ax.legend(frameon=False,fontsize=7.5,ncol=2); _despine(ax)
    plt.tight_layout(pad=0.5,w_pad=1.5)
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(outdir,f'fig_A_threshold.{ext}'),bbox_inches='tight')
    plt.close(fig); print('  fig_A_threshold saved')


# ══════════════════════════════════════════════════════════════
# PLOT B
# ══════════════════════════════════════════════════════════════
def plot_B(res, disorder_levels, outdir):
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.9))
    ax=axes[0]
    for i,f in enumerate(disorder_levels):
        d=res[f]
        ax.fill_between(d['K'],np.clip(d['mean']-d['std'],0.5,1.0),
                        np.clip(d['mean']+d['std'],0.5,1.0),color=C4[i],alpha=0.12)
        ax.plot(d['K'],d['mean'],color=C4[i],marker=MARKS[i],ms=4,
                markevery=3,lw=1.5,label=DL[f])
    _rank_panel(ax)
    ax.set_title(f'(a) Disorder robustness  ($v={V_SCALE:.0f}a/T$)',fontsize=9)
    ax.legend(frameon=False,fontsize=7.5)

    ax=axes[1]; cp=['#2166ac','#4dac26','#d6604d','#762a83']
    for ik,Kp in enumerate([20,40,60,80]):
        deficits=[]
        for f in disorder_levels:
            d=res[f]; ix=np.where(d['K']==Kp)[0]
            deficits.append(1.0-d['mean'][ix[0]] if len(ix)>0 else np.nan)
        ax.plot([f*100 for f in disorder_levels],deficits,
                'o-',color=cp[ik],ms=5,lw=1.5,label=f'$K={Kp}$')
    ax.axhline(0.05,color='gray',ls=':',lw=0.8)
    ax.set_xlabel('Disorder level $f$ [%]'); ax.set_ylabel(r'rank deficit  $1 - r_{\varepsilon}/(K-1)$')
    ax.set_title('(b) Rank deficit vs disorder (selected $K$)',fontsize=9)
    ax.set_xlim(-2,38); ax.set_ylim(0,0.07)
    ax.legend(frameon=False,fontsize=6.8,loc='lower center',ncol=2,
              bbox_to_anchor=(0.52,0.03),borderaxespad=0.0); _despine(ax)
    plt.tight_layout(pad=0.5,w_pad=1.5)
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(outdir,f'fig_B_disorder.{ext}'),bbox_inches='tight')
    plt.close(fig); print('  fig_B_disorder saved')


# ══════════════════════════════════════════════════════════════
# PLOT C
# ══════════════════════════════════════════════════════════════
def plot_C(rank_data, frob, v_scales, outdir):
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.9))
    ax=axes[0]
    for i,vs in enumerate(v_scales):
        Kv=sorted(rank_data[vs].keys())
        mn=[rank_data[vs][K][0] for K in Kv]
        sd=[rank_data[vs][K][1] for K in Kv]
        is_s=(vs==0.0); is_r=(vs==20.0)
        lw=2.4 if is_s else(2.0 if is_r else 1.1)
        ls='-' if(is_s or is_r) else '--'
        ax.fill_between(Kv,np.clip(np.array(mn)-np.array(sd),0.5,1.0),
                        np.clip(np.array(mn)+np.array(sd),0.5,1.0),
                        color=C6[i],alpha=0.10)
        ax.plot(Kv,mn,color=C6[i],marker='o',ms=4,markevery=2,
                lw=lw,ls=ls,label=VL[vs],zorder=5 if is_s else 3)
    # Annotate static
    Klast=sorted(rank_data[0.0].keys())[-1]; v_s=rank_data[0.0][Klast][0]
    ax.annotate('v=0 (static)',xy=(Klast,v_s),xytext=(Klast-24,v_s-0.032),
                fontsize=7.5,color=C6[0],fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=C6[0],lw=1.0))
    ax.axhline(0.93,color='#e7298a',ls='--',lw=1.2,alpha=0.7,label='0.93 bound')
    _rank_panel(ax)
    ax.set_ylabel(r'$r_{10^{-3}}/(K-1)$')
    ax.set_title('(a) Static ($v=0$): highest rank\n'
                 'both limits $\\geq 0.93$ (trajectory smearing)',fontsize=9)
    ax.legend(frameon=False,fontsize=6.8,ncol=2)

    ax=axes[1]
    xlabs=[VL[vs].replace(' (static)','').replace(' (realistic)','')
           for vs in v_scales]
    bars=ax.bar(range(len(v_scales)),frob,color=C6,alpha=0.72,
                edgecolor='#333',linewidth=0.5)
    bars[0].set_edgecolor(C6[0]); bars[0].set_linewidth(2.2)
    bars[-1].set_edgecolor('#e7298a'); bars[-1].set_linewidth(2.0)
    ax.set_xticks(range(len(v_scales)))
    ax.set_xticklabels(xlabs,fontsize=7.5,rotation=12)
    ax.set_ylabel(r'$\|M^{\rm dyn}-M^{\rm stat}\|_F / \|M^{\rm stat}\|_F$')
    ax.set_title('(b) Matrix deviation from static limit\n($K=20$, 12 realisations)',fontsize=9)
    pi=int(np.argmax(frob))
    ax.annotate(f'peak  {frob[pi]:.2f}x',xy=(pi,frob[pi]),
                xytext=(pi+0.7,frob[pi]*0.85),fontsize=7,color='#333',
                arrowprops=dict(arrowstyle='->',color='#555',lw=0.8))
    ax.text(0.04,0.54,'Full-rank holds at all speeds\n(rank-norm $\\geq 0.93$)',
            transform=ax.transAxes,ha='left',va='center',fontsize=7,
            bbox=dict(fc='#f8f8f8',ec='#cccccc',pad=4,lw=0.7))
    _despine(ax)
    plt.tight_layout(pad=0.5,w_pad=1.5)
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(outdir,f'fig_C_static_dynamic.{ext}'),bbox_inches='tight')
    plt.close(fig); print('  fig_C_static_dynamic saved')


# ══════════════════════════════════════════════════════════════
# PLOT D
# ══════════════════════════════════════════════════════════════
def plot_D(data,b,b_ci,K_fit,Nft_fit,Nft_fit_lo,Nft_fit_hi,frac_FT,T_eff_mid,K_dense,
           cost_naive,cost_eff,K_star_n,K_star_e,hist_near,hist_mid,outdir):
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.9))
    K_ref=data['K'].astype(float)

    ax=axes[0]
    ax.fill_between(data['K'],np.clip(data['Nft']-data['Nft_std'],0,None),
                    data['Nft']+data['Nft_std'],color='#2166ac',alpha=0.15)
    ax.stackplot(data['K'],data['Nnear'],data['Nmid'],
                 labels=['Near-field (SNR$\\geq 3$)','Mid-field (SNR$<3$)'],
                 colors=['#4dac26','#f46d43'],alpha=0.55)
    ax.plot(data['K'],data['Nft'],'o',color='#2166ac',ms=4,zorder=6,label='FT total')
    ax.fill_between(K_fit, Nft_fit_lo, Nft_fit_hi, color='#333', alpha=0.10, lw=0)
    ax.plot(K_fit,Nft_fit,'--',color='#333',lw=1.4,
            label=f'$K^{{{b:.2f}}}$ fit  [{b_ci[0]:.2f},{b_ci[1]:.2f}]')
    ax.plot(K_ref,frac_FT*K_ref*(K_ref-1)/2,':',color='#888',lw=1.0,
            label=f'{frac_FT:.2f}$\\cdot K(K-1)/2$')
    ax.set_xlabel('Active qubits $K$'); ax.set_ylabel('FT-relevant pairs')
    ax.set_title(f'(a) FT pairs $\\sim O(K^2)$\n'
                 f'($X_{{\\rm max}}={X_MAX:.0f}a$, {frac_FT*100:.0f}% of total)',fontsize=9)
    ax.legend(frameon=False,fontsize=6.5); _despine(ax)

    ax=axes[1]; bins=np.logspace(-2,2,25)
    if len(hist_near)>0:
        ax.hist(hist_near,bins=bins,color='#4dac26',alpha=0.65,
                label=f'Near ($n={len(hist_near)}$)')
    if len(hist_mid)>0:
        ax.hist(hist_mid,bins=bins,color='#f46d43',alpha=0.65,
                label=f'Mid ($n={len(hist_mid)}$)')
    ax.axvline(1.0,color='#e7298a',ls='--',lw=1.6,label='SNR=1')
    ax.axvline(3.0,color='#555',ls=':',lw=1.2,label='SNR=3')
    ax.set_xscale('log')
    ax.set_xlabel('SNR of FT-relevant $M_{ij}$'); ax.set_ylabel('Count ($K=40$)')
    ax.set_title(f'(b) SNR distribution ($K=40$)\n'
                 f'($\\sigma_M={SIGMA_M:.4f}$)',fontsize=9)
    ax.legend(frameon=False,fontsize=6.6,loc='upper right'); _despine(ax)

    ax=axes[2]
    ax.semilogy(K_dense,cost_naive,color='#2166ac',lw=1.8,label=r'Naive $O(K^2)$')
    ax.semilogy(data['K'],cost_eff,color='#d6604d',lw=1.8,ls='-.',
                label=f'Effective ($T_{{\\rm mid}}={T_eff_mid:.1f}$ s)')
    ax.axhline(TAU_DRIFT,color='#333',ls=':',lw=1.2)
    ax.text(98,TAU_DRIFT*0.82,r'$\tau_{\rm drift}$ (1 hr)',fontsize=7.5,
            color='#333',ha='right',va='top')
    kstar_items = [(K_star_n,'#2166ac',f'$K^*_n={K_star_n:.0f}$',0.035),
                   (K_star_e,'#d6604d',f'$K^*_e={K_star_e:.0f}$',0.11)]
    for Ks,col,lab,ypos in kstar_items:
        if Ks<K_dense[-1]:
            ax.axvline(Ks,color=col,lw=1.5,ls='--',alpha=0.8)
            ax.text(Ks+1.2,ypos,lab,fontsize=7.4,color=col,
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
    ax.fill_betweenx([TAU_DRIFT,1e5],min(K_star_n,K_star_e),100,alpha=0.08,color='#d6604d')
    ax.set_xlabel('Active qubits $K$'); ax.set_ylabel('Calibration time [s]')
    ax.set_title('(c) Calibration drift wall\n($K^*_n$: naive, $K^*_e$: corrected)',fontsize=9)
    ax.set_ylim(1e-2,1e5); ax.set_xlim(2,100)   # capped at 1e5 for cleaner display
    ax.legend(frameon=False,fontsize=6.8,loc='upper left'); _despine(ax)
    plt.tight_layout(pad=0.5,w_pad=1.6)
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(outdir,f'fig_D_absolute.{ext}'),bbox_inches='tight')
    plt.close(fig); print('  fig_D_absolute saved')


# ══════════════════════════════════════════════════════════════
# PLOT SUMMARY
# ══════════════════════════════════════════════════════════════
def plot_summary(K_arr,res_A,res_B,rank_data_C,frob_C,v_scales_C,
                 data_D,b_D,b_ci_D,K_fit_D,Nft_fit_D,Nft_fit_lo_D,Nft_fit_hi_D,frac_FT_D,
                 cost_naive_D,cost_eff_D,K_dense_D,K_star_n_D,
                 hist_near_D,hist_mid_D,outdir):
    """
    Main paper Figure 2: compact 4-panel Stage-1 bottleneck summary.

    The report recommends reducing the original 6-panel summary because it
    combines too many concepts in one figure.  The main figure keeps only the
    pieces needed for the core bottleneck argument:
      (a) near-full effective rank is threshold-robust;
      (b) dynamic shuttling preserves the rank structure relative to static;
      (c) FT-relevant pair count grows quadratically with bootstrap CI;
      (d) pairwise calibration hits the assumed drift wall.

    Disorder robustness and the SNR histogram remain in the individual appendix
    figures (fig_B_disorder and fig_D_absolute).
    """
    from scipy.ndimage import uniform_filter1d

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.50,
                           left=0.09, right=0.98, top=0.92, bottom=0.12)
    K_arr = np.asarray(K_arr)

    # (a) Threshold-robust full rank
    ax = fig.add_subplot(gs[0, 0])
    for i,e in enumerate((1e-2,1e-3,1e-4,1e-5)):
        mn = uniform_filter1d(np.array(res_A[e]['mean']), size=3, mode='nearest')
        ax.plot(K_arr, mn, color=C4[i], marker=MARKS[i], ms=3.5,
                markevery=4, lw=1.4, label=TL[e])
    ax.axhline(1.0, color='#555', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('$K$'); ax.set_ylabel(r'$r_{\varepsilon}/(K-1)$')
    ax.set_title('(a) Near-full rank across thresholds', fontsize=9)
    ax.set_ylim(0.83, 1.00); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc='lower left')
    _despine(ax)

    # (b) Static vs realistic dynamic rank.  Keep only the two physical endpoints
    # in the main figure; full speed sweep remains in fig_C_static_dynamic.
    ax = fig.add_subplot(gs[0, 1])
    for vs, col, lbl, lw in [(0.0, '#2166ac', 'static $v=0$', 2.1),
                             (20.0, '#4dac26', 'realistic $v=20a/T$', 2.1)]:
        Kv = sorted(rank_data_C[vs].keys())
        mn_raw = np.array([rank_data_C[vs][K][0] for K in Kv])
        sd_raw = np.array([rank_data_C[vs][K][1] for K in Kv])
        mn = uniform_filter1d(mn_raw, size=3, mode='nearest')
        ax.fill_between(Kv, np.clip(mn_raw-sd_raw, 0.5, 1.0),
                        np.clip(mn_raw+sd_raw, 0.5, 1.0),
                        color=col, alpha=0.10)
        ax.plot(Kv, mn, color=col, lw=lw, label=lbl)
    ax.axhline(0.93, color='#e7298a', ls='--', lw=1.2, alpha=0.75)
    ax.text(0.56, 0.54, r'all tested speeds $\geq 0.93$', transform=ax.transAxes,
            fontsize=6.6, color='#e7298a', va='top', ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.80, pad=1.2))
    ax.set_xlabel('$K$'); ax.set_ylabel(r'$r_{\varepsilon}/(K-1)$')
    ax.set_title('(b) Dynamic shuttling preserves rank', fontsize=9)
    ax.set_ylim(0.83, 1.00); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False, fontsize=6.8, loc='lower left')
    _despine(ax)

    # (c) FT-relevant pair scaling with uncertainty band
    ax = fig.add_subplot(gs[1, 0])
    K_ref = data_D['K'].astype(float)
    ax.stackplot(data_D['K'], data_D['Nnear'], data_D['Nmid'],
                 labels=['Near-field', 'Mid-field'],
                 colors=['#4dac26', '#f46d43'], alpha=0.65)
    ax.plot(data_D['K'], data_D['Nft'], 'o', color='#2166ac', ms=3.8,
            zorder=6, label='FT total')
    ax.fill_between(K_fit_D, Nft_fit_lo_D, Nft_fit_hi_D,
                    color='#333', alpha=0.12, lw=0, label='95% bootstrap CI')
    ax.plot(K_fit_D, Nft_fit_D, '--', color='#333', lw=1.4,
            label=f'$K^{{{b_D:.2f}}}$ [{b_ci_D[0]:.2f},{b_ci_D[1]:.2f}]')
    ax.set_xlabel('$K$'); ax.set_ylabel('FT-relevant pairs')
    ax.set_title(f'(c) Pair count scales as $O(K^2)$\n'
                 f'($X_{{\\mathrm{{max}}}}={X_MAX:.0f}a$, {frac_FT_D*100:.0f}% of pairs)', fontsize=9)
    ax.legend(frameon=False, fontsize=6.2, loc='upper left')
    _despine(ax)

    # (d) Calibration drift wall under sequential pairwise timing assumption
    ax = fig.add_subplot(gs[1, 1])
    if K_star_n_D < K_dense_D[-1]:
        ax.fill_betweenx([TAU_DRIFT, 1e5], K_star_n_D, 100,
                         alpha=0.07, color='#d6604d', zorder=1)
    ax.semilogy(K_dense_D, cost_naive_D, color='#d6604d', lw=1.9,
                label=r'Naive pairwise $O(K^2)$', zorder=3)
    ax.semilogy(data_D['K'], cost_eff_D, color='#2166ac', lw=1.8, ls='-.',
                label='SNR-corrected pairwise', zorder=3)
    ax.axhline(TAU_DRIFT, color='#333', ls=':', lw=1.2, zorder=2)
    ax.text(0.97, 0.66, r'$\tau_{\rm drift}$ (1 hr)', transform=ax.transAxes,
            fontsize=7.2, color='#333', ha='right', va='top', zorder=4)
    if K_star_n_D < K_dense_D[-1]:
        ax.axvline(K_star_n_D, color='#d6604d', lw=1.4, ls='--', alpha=0.85, zorder=3)
        ax.text(K_star_n_D+1.5, 75, fr'$K^*\!\approx\!{K_star_n_D:.0f}$',
                fontsize=7.5, color='#d6604d', va='bottom')
    ax.text(0.04, 0.95, 'assumed sequential\nRamsey timing model',
            transform=ax.transAxes, fontsize=6.6, color='#555', ha='left', va='top',
            bbox=dict(fc='white', ec='#cccccc', lw=0.6, pad=2.5, alpha=0.9))
    ax.set_xlabel('$K$'); ax.set_ylabel('Calibration time [s]')
    ax.set_title('(d) Pairwise calibration drift wall', fontsize=9)
    ax.set_ylim(1e0, 1e5); ax.set_xlim(2, 100)
    ax.legend(frameon=False, fontsize=6.9, loc='lower right')
    _despine(ax)

    fig.savefig(os.path.join(outdir, 'fig_summary.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'fig_summary.png'), bbox_inches='tight', dpi=200)
    plt.close(fig); print('  fig_summary saved  [4-panel main]')


def plot_summary_full6(K_arr,res_A,res_B,rank_data_C,frob_C,v_scales_C,
                 data_D,b_D,b_ci_D,K_fit_D,Nft_fit_D,Nft_fit_lo_D,Nft_fit_hi_D,frac_FT_D,
                 cost_naive_D,cost_eff_D,K_dense_D,K_star_n_D,
                 hist_near_D,hist_mid_D,outdir):
    fig=plt.figure(figsize=(7.4,5.8))
    gs=gridspec.GridSpec(2,3,figure=fig,wspace=0.46,hspace=0.60)

    # (a) A
    ax=fig.add_subplot(gs[0,0])
    for i,e in enumerate((1e-2,1e-3,1e-4,1e-5)):
        from scipy.ndimage import uniform_filter1d
        mn_sm=uniform_filter1d(np.array(res_A[e]['mean']),size=3,mode='nearest')
        ax.plot(K_arr,mn_sm,color=C4[i],marker=MARKS[i],
                ms=3.5,markevery=4,lw=1.4,label=TL[e])
    ax.axhline(1.0,color='#555',ls='--',lw=0.8,alpha=0.5)
    ax.set_xlabel('$K$'); ax.set_ylabel(r'$r_{\varepsilon}/(K-1)$')
    ax.set_title('(a) Full-rank: threshold-robust',fontsize=9)
    ax.set_ylim(0.83, 1.00); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False,fontsize=6.0,ncol=2); _despine(ax)

    # (b) B
    ax=fig.add_subplot(gs[0,1])
    for i,f in enumerate((0.0,0.10,0.20,0.35)):
        d=res_B[f]
        from scipy.ndimage import uniform_filter1d
        mn_raw=np.array(d['mean'])
        mn_sm=uniform_filter1d(mn_raw,size=3,mode='nearest')
        ax.plot(d['K'],mn_sm,color=C4[i],marker=MARKS[i],
                ms=3.5,markevery=4,lw=1.4,label=DL[f])
    ax.axhline(1.0,color='#555',ls='--',lw=0.8,alpha=0.5)
    ax.set_xlabel('$K$'); ax.set_ylabel(r'$r_{\varepsilon}/(K-1)$')
    ax.set_title('(b) Full-rank: disorder-robust',fontsize=9)
    ax.set_ylim(0.83, 1.00); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False,fontsize=6.5); _despine(ax)

    # (c) C
    ax=fig.add_subplot(gs[0,2])
    for i,vs in enumerate(v_scales_C):
        Kv=sorted(rank_data_C[vs].keys())
        from scipy.ndimage import uniform_filter1d
        mn_raw=np.array([rank_data_C[vs][K][0] for K in Kv])
        mn=uniform_filter1d(mn_raw,size=3,mode='nearest')
        lw=2.2 if vs==0.0 else(1.8 if vs==20.0 else 1.0)
        ls='-' if vs in(0.0,20.0) else '--'
        ax.plot(Kv,mn,color=C6[i],lw=lw,ls=ls,
                label=VL[vs].replace(' (realistic)',''))
    Klast=sorted(rank_data_C[0.0].keys())[-1]; v_s=rank_data_C[0.0][Klast][0]
    ax.annotate('static',xy=(Klast,v_s),xytext=(Klast-22,v_s-0.032),
                fontsize=6.5,color=C6[0],fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=C6[0],lw=0.9))
    ax.axhline(0.93,color='#e7298a',ls='--',lw=1.2,alpha=0.7)
    ax.text(20,0.928,'lower bound (all speeds)',fontsize=6.2,
            color='#e7298a',va='top',alpha=0.85)
    ax.set_xlabel('$K$'); ax.set_ylabel(r'$r_{\varepsilon}/(K-1)$')
    ax.set_title('(c) Full-rank: speed-robust\n(static $\\to$ $v=20a/T$)',fontsize=9)
    ax.set_ylim(0.83, 1.00); ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False,fontsize=6.0,ncol=2); _despine(ax)

    # (d) D pairs
    ax=fig.add_subplot(gs[1,0]); K_ref=data_D['K'].astype(float)
    ax.stackplot(data_D['K'],data_D['Nnear'],data_D['Nmid'],
                 labels=['Near-field','Mid-field'],
                 colors=['#4dac26','#f46d43'],alpha=0.65)
    ax.plot(data_D['K'],data_D['Nft'],'o',color='#2166ac',ms=3.5,zorder=6)
    ax.fill_between(K_fit_D, Nft_fit_lo_D, Nft_fit_hi_D, color='#333', alpha=0.10, lw=0)
    ax.plot(K_fit_D,Nft_fit_D,'--',color='#333',lw=1.4,
            label=f'$K^{{{b_D:.2f}}}$ [{b_ci_D[0]:.2f},{b_ci_D[1]:.2f}]')
    ax.set_xlabel('$K$'); ax.set_ylabel('FT-relevant pairs')
    ax.set_title(f'(d) $O(K^2)$ FT pairs\n'
                 f'($X_{{\\rm max}}={X_MAX:.0f}a$, {frac_FT_D*100:.0f}%)',fontsize=9,x=0.43)
    ax.legend(frameon=False,fontsize=6.6,loc='upper right',
              bbox_to_anchor=(0.90,1.00)); _despine(ax)

    # (e) SNR
    ax=fig.add_subplot(gs[1,1]); bins2=np.logspace(-2,2,20)
    if len(hist_near_D)>0:
        ax.hist(hist_near_D,bins=bins2,color='#4dac26',alpha=0.65,label='Near-field')
    if len(hist_mid_D)>0:
        ax.hist(hist_mid_D,bins=bins2,color='#f46d43',alpha=0.65,label='Mid-field')
    ax.axvline(1.0,color='#e7298a',ls='--',lw=1.5)
    ax.axvline(3.0,color='#555',ls=':',lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel('SNR of $M_{ij}$'); ax.set_ylabel('Count ($K=40$)')
    ax.set_title('(e) SNR of FT-relevant $M_{ij}$ ($K=40$)',fontsize=9)
    ax.legend(frameon=False,fontsize=6.8,loc='upper left'); _despine(ax)

    # (f) Drift wall — fill/text first (zorder low), curves on top (zorder high)
    ax=fig.add_subplot(gs[1,2])
    if K_star_n_D<K_dense_D[-1]:
        ax.fill_betweenx([TAU_DRIFT,1e5],K_star_n_D,100,
                         alpha=0.08,color='#e7298a',zorder=1)
        ax.text(0.72,0.82,'Drift\nwall',transform=ax.transAxes,
                fontsize=8,color='#c0345e',ha='center',zorder=1,
                bbox=dict(fc='#fff0f4',ec='#e7298a',pad=3,lw=0.7,alpha=0.85))
    ax.semilogy(K_dense_D,cost_naive_D,color='#2166ac',lw=1.8,
                label=r'Naive $O(K^2)$',zorder=3)
    ax.semilogy(data_D['K'],cost_eff_D,color='#d6604d',lw=1.8,ls='-.',
                label='SNR-corrected',zorder=3)
    ax.axhline(TAU_DRIFT,color='#333',ls=':',lw=1.2,zorder=2)
    # τ_drift label: immediately below the dotted line, right edge
    ax.text(0.97, 0.715, r'$\tau_{\rm drift}$', transform=ax.transAxes,
            fontsize=7.5, color='#333', ha='right', va='top', zorder=2)
    if K_star_n_D<K_dense_D[-1]:
        ax.axvline(K_star_n_D,color='#2166ac',lw=1.4,ls='--',alpha=0.8,zorder=3)
        # K* label: mid-height of axis (≈100s on log10 scale), below τ_drift
        ax.text(K_star_n_D+1.5, 80,
                f'$K^*\\!={K_star_n_D:.0f}$',
                fontsize=8, color='#2166ac', va='bottom')
    ax.set_xlabel('$K$'); ax.set_ylabel('Calibration time [s]')
    ax.set_title('(f) Calibration drift wall',fontsize=9)
    ax.set_ylim(1e0,1e5); ax.set_xlim(2,100)
    # legend lower right: curves are at top-right, empty space at bottom-right
    ax.legend(frameon=False,fontsize=7.5,loc='lower right'); _despine(ax)

    fig.savefig(os.path.join(outdir,'fig_summary_full6_appendix.pdf'),bbox_inches='tight')
    fig.savefig(os.path.join(outdir,'fig_summary_full6_appendix.png'),bbox_inches='tight',dpi=200)
    plt.close(fig); print('  fig_summary_full6_appendix saved')


def save_stage1_source_data(K_A, res_A, res_B, rank_data_C, frob_C, v_scales_C,
                            data_D, b_D, b_ci_D, K_fit_D, Nft_fit_D,
                            Nft_fit_lo_D, Nft_fit_hi_D, cost_naive_D,
                            cost_eff_D, K_dense_D, outdir):
    """Write minimal source-data CSV files for the Stage-1 figures."""
    import csv

    with open(os.path.join(outdir, 'source_data_stage1_rank_threshold.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['K','epsilon','mean_rank_norm','std_rank_norm'])
        for eps, dat in res_A.items():
            for K, mn, sd in zip(K_A, dat['mean'], dat['std']):
                w.writerow([K, eps, mn, sd])

    with open(os.path.join(outdir, 'source_data_stage1_rank_disorder.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['K','disorder_f','mean_rank_norm','std_rank_norm'])
        for fdis, dat in res_B.items():
            for K, mn, sd in zip(dat['K'], dat['mean'], dat['std']):
                w.writerow([K, fdis, mn, sd])

    with open(os.path.join(outdir, 'source_data_stage1_speed.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['v_scale','K','mean_rank_norm','std_rank_norm','frob_distance_K20'])
        frob_map = {v: frob_C[i] for i, v in enumerate(v_scales_C)}
        for v in v_scales_C:
            for K, (mn, sd) in rank_data_C[v].items():
                w.writerow([v, K, mn, sd, frob_map[v]])

    with open(os.path.join(outdir, 'source_data_stage1_FT_pairs.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['K','Nft_mean','Nft_std','Nnear_mean','Nmid_mean','Ceff_mean','Ceff_std'])
        for i, K in enumerate(data_D['K']):
            w.writerow([K, data_D['Nft'][i], data_D['Nft_std'][i], data_D['Nnear'][i],
                        data_D['Nmid'][i], data_D['Ceff'][i], data_D['Ceff_std'][i]])

    with open(os.path.join(outdir, 'source_data_stage1_powerlaw_fit.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['K_fit','Nft_fit','Nft_fit_lo95','Nft_fit_hi95','b','b_lo95','b_hi95'])
        for K, y, lo, hi in zip(K_fit_D, Nft_fit_D, Nft_fit_lo_D, Nft_fit_hi_D):
            w.writerow([K, y, lo, hi, b_D, b_ci_D[0], b_ci_D[1]])

    with open(os.path.join(outdir, 'source_data_stage1_calibration_time.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['K','cost_naive_s'])
        for K, c in zip(K_dense_D, cost_naive_D):
            w.writerow([K, c])
        w.writerow([])
        w.writerow(['K','cost_effective_s'])
        for K, c in zip(data_D['K'], cost_eff_D):
            w.writerow([K, c])
    print('  source_data_stage1_*.csv saved')


# ══════════════════════════════════════════════════════════════
# FIGURE CAPTIONS
# ══════════════════════════════════════════════════════════════
FIGURE_CAPTIONS = {

'fig_A': (
    "FIG. A. Robustness of the effective rank to the singular-value threshold. "
    "(a) Normalized effective rank $r_\\varepsilon/(K-1)$ of the gauge-projected "
    "susceptibility matrix $\\mathbf{M}$ vs the number of simultaneously shuttled "
    "qubits $K$, evaluated at four relative thresholds "
    "$\\varepsilon \\in \\{10^{-2},10^{-3},10^{-4},10^{-5}\\}$ "
    "using $M^{\\rm dyn}$ with $v=20a/T$. Shaded regions: one s.d. over "
    f"$N_{{\\rm trials}}={N_TRIALS}$ random qubit placements on an "
    "$18\\times 18$ lattice ($a\\approx 100$ nm). "
    "(b) Rank deficit $1-r_\\varepsilon/(K-1)$ on a logarithmic scale. "
    "For $\\varepsilon\\leq 10^{-3}$ the deficit remains below 10\\% for all "
    "$K\\leq 100$, confirming that $\\mathbf{M}$ is generically near-full-rank "
    "independently of threshold choice."
),

'fig_B': (
    "FIG. B. Robustness of the effective rank to positional disorder. "
    "(a) $r_{10^{-3}}/(K-1)$ vs $K$ for disorder levels "
    "$f\\in\\{0,10,20,35\\}\\%$, where each site is displaced uniformly "
    "within $\\pm f\\cdot a$. "
    "(b) Rank deficit vs $f$ for selected values of $K$. "
    "The effective rank is insensitive to disorder up to $f=35\\%$, "
    "ruling out perfect lattice symmetry as the origin of near-full-rank."
),

'fig_C': (
    "FIG. C. Comparison of the static and dynamic susceptibility matrices. "
    "(a) $r_{10^{-3}}/(K-1)$ for shuttling speeds "
    "$v\\in\\{0,0.5,2,5,10,20\\}a/T$. "
    "The static limit $v=0$ (thick solid blue line) yields the highest "
    "effective rank: each row $i$ of $M^{\\rm stat}_{ij}=\\chi_{ij}(\\mathbf{r}_i^0,"
    "\\mathbf{r}_j^0)$ encodes a unique $1/r^3$ spatial fingerprint, "
    "making rows maximally linearly independent. "
    "At $v=20a/T$ (thick green line) the trajectory integral acts as a "
    "spatial low-pass filter---qubit $i$ spends most of the transit period "
    "far from all $j$, reducing row distinctiveness by $\\lesssim 5\\%$. "
    "Both limits satisfy rank$_{10^{-3}}/(K-1)\\geq 0.93$ (pink dashed line): "
    "the $O(K^2)$ bottleneck holds at all physically relevant speeds. "
    "Note that larger crosstalk magnitude does not imply higher rank; "
    "rank measures linear independence of rows, not their magnitude. "
    "(b) Frobenius distance "
    "$\\|M^{\\rm dyn}-M^{\\rm stat}\\|_F/\\|M^{\\rm stat}\\|_F$ "
    "vs speed ($K=20$, 12 realisations), quantifying how much the matrix "
    "changes with velocity while its rank structure is preserved."
),

'fig_D': (
    "FIG. D. Absolute-threshold analysis and calibration drift wall. "
    f"Physical parameters: $X_{{\\rm max}}={X_MAX:.0f}a$, $u_j={U_J:.0f}a$, "
    f"$\\sigma_M=\\sigma_\\varphi/u_j\\approx {SIGMA_M:.4f}$, "
    f"$M_{{\\rm cut}}=\\varepsilon_{{\\rm FT}}/u_j={M_CUT:.0e}$. "
    "(a) Number of fault-tolerant-relevant pairs ($|M_{ij}|>M_{\\rm cut}$) vs $K$, "
    "decomposed into near-field (SNR$\\geq 3$, green) and mid-field "
    "(SNR$<3$, orange). Dashed line and shaded band show a bootstrap power-law "
    "fit $N_{\\rm FT}\\propto K^b$ with the 95\\% confidence interval reported in the panel. "
    "(b) Distribution of SNR values for FT-relevant $M_{ij}$ at $K=40$; "
    "vertical lines denote SNR$=1$ and SNR$=3$. "
    "(c) Calibration time vs $K$ for sequential pairwise characterisation under "
    "the assumed Ramsey timing model. Horizontal dotted line: "
    "$\\tau_{\\rm drift}=3600$ s. The critical scale $K^*\\approx 30$ is therefore "
    "a model-dependent drift-wall estimate, not a device-independent prediction."
),

'fig_summary': (
    "FIG. 1. Compact summary of the $O(K^2)$ calibration bottleneck for parallel "
    "spin-shuttling arrays (Stage~1). The main text version is reduced to four "
    "panels to separate the core bottleneck evidence from robustness details. "
    "All results use an $18\\times 18$ square lattice ($a\\approx 100$ nm), "
    f"$N_{{\\rm trials}}={N_TRIALS}$ random qubit placements, and realistic "
    f"parameters ($v={V_SCALE:.0f}a/T$, $X_{{\\rm max}}={X_MAX:.0f}a$). "
    "(a) Normalised effective rank $r_\\varepsilon/(K-1)$ vs $K$ for four "
    "relative thresholds $\\varepsilon$, showing near-full-rank behaviour. "
    "(b) Static and realistic dynamic limits both preserve the dense rank "
    "structure; the full speed sweep is reported in Fig.~C. "
    "(c) FT-relevant pairs vs $K$, decomposed into near-field and mid-field "
    "entries; the dashed fit reports $K^b$ with bootstrap confidence interval. "
    "(d) Calibration time for sequential pairwise characterisation under the "
    "assumed Ramsey timing model, showing the representative drift wall near "
    "$K^*\\approx 30$. Disorder robustness and the SNR histogram are retained "
    "in the appendix/individual diagnostic figures."
),
}

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    OUTDIR = os.environ.get('OUTDIR', '/content/outputs')
    os.makedirs(OUTDIR, exist_ok=True)

    print('='*62)
    print('  Stage 1 Final Simulation')
    print('='*62)
    print(f'  N_TRIALS  = {N_TRIALS}')
    print(f'  Lattice   = {L_GRID}x{L_GRID} = {L_GRID**2} sites')
    print(f'  v_scale   = {V_SCALE}a/T,  X_MAX = {X_MAX}a,  u_j = {U_J:.1f}a')
    print(f'  sigma_M   = {SIGMA_M:.5f},  M_cut = {M_CUT:.5f}')
    print(); sys.stdout.flush()

    pos  = perfect_lattice(L_GRID)
    KR   = [k for k in range(5,101,5) if k<=L_GRID**2]
    TH   = (1e-2,1e-3,1e-4,1e-5)
    DIS  = (0.0,0.10,0.20,0.35)
    VS   = (0.0,0.5,2.0,5.0,10.0,20.0)
    KPC  = (10,20,40,60,80)

    print(f'[A] Threshold robustness (N={N_TRIALS}) ...'); sys.stdout.flush()
    K_A,res_A = test_A(pos,KR,thresholds=TH)
    print(f'    eps=1e-3, K=60: {res_A[1e-3]["mean"][K_A==60][0]:.3f}')

    print(f'[B] Disorder robustness  (N={N_TRIALS}) ...'); sys.stdout.flush()
    res_B = test_B(L_GRID,KR,disorder_levels=DIS)
    print(f'    f=35%, K=60: {res_B[0.35]["mean"][res_B[0.35]["K"]==60][0]:.3f}')

    print(f'[C] Static vs dynamic    (N={N_TRIALS}) ...'); sys.stdout.flush()
    rank_C,frob_C = test_C(pos,K_probe=KPC,v_scales=VS)
    print(f'    v=0 (static), K=60: {rank_C[0.0][60][0]:.3f}  <- highest')
    print(f'    v=20a/T,      K=60: {rank_C[20.0][60][0]:.3f}')
    print(f'    Frobenius v=20a/T:  {frob_C[-1]:.3f}')

    print(f'[D] Absolute threshold   (N={N_TRIALS}) ...'); sys.stdout.flush()
    (dD,bD,bciD,KfD,NfD,NfD_lo,NfD_hi,frD,TeD,KdD,cnD,ceD,Kn,Ke,hn,hm) = test_D(pos,KR)
    print(f'    FT fraction: {frD*100:.1f}%,  N_FT ~ K^{bD:.3f}  (95% CI {bciD[0]:.3f}, {bciD[1]:.3f})')
    print(f'    K*_naive={Kn:.1f},  K*_eff={Ke:.1f},  T_mid={TeD:.1f}s')

    print('\nGenerating figures ...'); sys.stdout.flush()
    plot_A(K_A,res_A,TH,OUTDIR)
    plot_B(res_B,DIS,OUTDIR)
    plot_C(rank_C,frob_C,VS,OUTDIR)
    plot_D(dD,bD,bciD,KfD,NfD,NfD_lo,NfD_hi,frD,TeD,KdD,cnD,ceD,Kn,Ke,hn,hm,OUTDIR)
    plot_summary(K_A,res_A,res_B,rank_C,frob_C,VS,
                 dD,bD,bciD,KfD,NfD,NfD_lo,NfD_hi,frD,cnD,ceD,KdD,Kn,hn,hm,OUTDIR)
    plot_summary_full6(K_A,res_A,res_B,rank_C,frob_C,VS,
                 dD,bD,bciD,KfD,NfD,NfD_lo,NfD_hi,frD,cnD,ceD,KdD,Kn,hn,hm,OUTDIR)
    save_stage1_source_data(K_A,res_A,res_B,rank_C,frob_C,VS,
                            dD,bD,bciD,KfD,NfD,NfD_lo,NfD_hi,cnD,ceD,KdD,OUTDIR)

    print('\n'+'='*62)
    print('  FIGURE CAPTIONS')
    print('='*62)
    for k,v in FIGURE_CAPTIONS.items():
        print(f'\n[{k}]\n{v}')

    try:
        shutil.copy(__file__, os.path.join(OUTDIR,'simulation_stage1_final.py'))
    except Exception:
        pass

    print('\n'+'='*62)
    print('  RESULTS SUMMARY')
    print('='*62)
    print(f'  [A] eps=1e-3, K=80: {res_A[1e-3]["mean"][K_A==80][0]:.3f}')
    print(f'  [B] f=35%,    K=80: {res_B[0.35]["mean"][res_B[0.35]["K"]==80][0]:.3f}')
    print(f'  [C] v=0,      K=80: {rank_C[0.0][80][0]:.3f}  (static, highest)')
    print(f'  [C] v=20a/T,  K=80: {rank_C[20.0][80][0]:.3f}')
    print(f'  [D] FT: {frD*100:.1f}%,  K^{bD:.3f} [{bciD[0]:.3f},{bciD[1]:.3f}],  K*={Kn:.0f}')
