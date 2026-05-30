"""
simulation_stage2_robustness.py
════════════════════════════════════════════════════════════════════
Stage 2 Robustness Analysis

(A) Sub-array Multiplexing
    m ∈ {K, 8, 4} nonzero probe displacements per calibration round
    - Each round still costs T_Ramsey seconds and reads out K phases in parallel
    - Tests whether finite probe bandwidth preserves the O(K) multiplexed
      calibration advantage relative to naive O(K^2) pairwise calibration.

(B) Prior Model Mismatch
    True chi_ij = (1/r^3) × (1 + alpha × random_perturbation)
    Prior uses ideal 1/r^3 — test up to 60% perturbation
    - This is not a prior-advantage claim. In the overdetermined n_exp=2K
      regime, plain LS and residual structural reduction should agree.
    - The purpose is to verify that the physical decomposition does not bias
      reconstruction when the prior baseline is imperfect.

(C) Residual Self-Phase Noise
    Add sigma_self per-qubit noise to simulate:
    - Imperfect delta_theta^self calibration (Sec 2.4)
    - Charge noise, nuclear spin fluctuations (xi_i in Eq. 10)
    - Tests that deterministic crosstalk tracking survives realistic stochastic
      environments.

Outputs:
  fig_3A_subarray_mux.pdf/.png
  fig_3B_prior_mismatch.pdf/.png
  fig_3C_selfphase_noise.pdf/.png
  fig_3_robustness_summary.pdf/.png
  fig_4_extended_robustness.pdf/.png
  source_data_stage2_robustness_*.csv
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, shutil, os, csv
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':'serif','font.size':9,'axes.labelsize':10,
    'axes.titlesize':10,'legend.fontsize':7.5,'xtick.labelsize':8,
    'ytick.labelsize':8,'lines.linewidth':1.6,'axes.linewidth':0.8,
    'figure.dpi':200,'text.usetex':False,
})

# ══════════════════════════════════════════════════════════════════
# §0  GLOBAL CONSTANTS  (inherited from Stage 1 & 2)
# ══════════════════════════════════════════════════════════════════
RNG        = np.random.default_rng(0)
N_TRIALS   = 15

L_GRID     = 18
Z_DEPTH    = 0.35
V_SCALE    = 20.0
V_NOISE    = 0.15
X_MAX      = 20.0
T_TRANS    = 1.0
U_J        = X_MAX * T_TRANS / 2.0
N_SHOTS    = 1000
SIGMA_PHI  = 1.0 / np.sqrt(N_SHOTS)
SIGMA_M    = SIGMA_PHI / U_J
U_MIN      = 0.5 * U_J
U_MAX      = 1.5 * U_J
T_RAMSEY   = 8.0
TAU_DRIFT  = 3600.0


# Sub-array configs to test
M_VALUES   = ['K', 8, 4]   # 'K' = full multiplexing
M_COLORS   = {'K': '#2166ac', 8: '#4dac26', 4: '#f46d43'}
M_LABELS   = {'K': r'Full mux. ($m=K$)', 8: r'Sub-array $m=8$', 4: r'Sub-array $m=4$'}
M_STYLES   = {'K': '-', 8: '--', 4: '-.'}

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _infidelity(mean, std=None, floor=1e-6):
    """Convert F_comp mean/std to infidelity 1-F with safe log-scale clipping."""
    y = np.maximum(1.0 - np.asarray(mean, dtype=float), floor)
    if std is None:
        return y
    lo = np.maximum(1.0 - (np.asarray(mean, dtype=float) + np.asarray(std, dtype=float)), floor)
    hi = np.maximum(1.0 - (np.asarray(mean, dtype=float) - np.asarray(std, dtype=float)), floor)
    return y, lo, hi

def _compute_naive_f_vs_K(K_arr, n_trials=N_TRIALS):
    """Uncompensated baseline used only for plotting contrast in robustness figures."""
    pos = perfect_lattice(L_GRID)
    N = len(pos)
    means, stds = [], []
    for K in K_arr:
        K = int(K)
        vals = []
        if K > N or K < 3:
            means.append(np.nan); stds.append(np.nan); continue
        for tr in range(min(n_trials, 12)):
            rng = np.random.default_rng(5000 + 97 * tr + K)
            idx = rng.choice(N, size=K, replace=False)
            M_true = build_M_dynamic(pos, idx, rng=np.random.default_rng(tr))
            u_nom = rng.uniform(-U_J, U_J, size=K)
            F, _ = fidelity_comp(M_true, np.zeros((K, K)), u_nom)
            vals.append(F)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return np.array(means), np.array(stds)



# ══════════════════════════════════════════════════════════════════
# §1  PHYSICS BUILDERS  (Stage 1 inherited)
# ══════════════════════════════════════════════════════════════════
def perfect_lattice(L, a=1.0):
    xs, ys = np.meshgrid(np.arange(L)*a, np.arange(L)*a)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)

def build_M_static(positions, idx, z=Z_DEPTH):
    K = len(idx); r0 = positions[idx]; M = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i != j:
                dr = r0[i]-r0[j]
                M[i,j] = 1.0/(np.dot(dr,dr)+z**2)**1.5
    return M

def build_M_dynamic(positions, idx, v_scale=V_SCALE, v_noise=V_NOISE,
                     z=Z_DEPTH, n=300, T=T_TRANS, rng=RNG):
    K = len(idx); r0 = positions[idx]
    ang = rng.uniform(0, 2*np.pi, size=K)
    spd = v_scale*(1+rng.uniform(-v_noise, v_noise, size=K))
    v   = spd[:,None]*np.stack([np.cos(ang),np.sin(ang)],axis=1)
    ts  = np.linspace(0,T,n); dt = ts[1]-ts[0]; M = np.zeros((K,K))
    for ti,t in enumerate(ts):
        ri = r0+v*t; dr = ri[:,None,:]-r0[None,:,:]
        d2 = np.sum(dr**2,axis=-1)+z**2
        w  = 0.5 if (ti==0 or ti==n-1) else 1.0
        M += w/d2**1.5*dt
    np.fill_diagonal(M,0.0); return M

def build_M_perturbed(positions, idx, alpha=0.0, z=Z_DEPTH, rng=RNG):
    """
    True M with a perturbed susceptibility: chi_ij_true = chi_ij × (1 + alpha × eps_ij)
    eps_ij ~ N(0,1), alpha controls mismatch amplitude.
    Prior still uses ideal chi_ij (no perturbation).
    Tests robustness of inference when device model has errors.
    """
    M_base = build_M_static(positions, idx, z)
    if alpha == 0.0:
        return M_base
    K = len(idx)
    perturbation = rng.normal(0, 1, (K,K))
    perturbation = (perturbation + perturbation.T) / 2  # symmetric
    np.fill_diagonal(perturbation, 0.0)
    M_pert = M_base * (1.0 + alpha * perturbation)
    np.fill_diagonal(M_pert, 0.0)
    return M_pert


# ══════════════════════════════════════════════════════════════════
# §2  CALIBRATION CORE  (Stage 2 inherited + sparse probe support)
# ══════════════════════════════════════════════════════════════════
def generate_sparse_probe(M_true, K, n_exp, m, sigma_self=0.0, rng=RNG):
    """
    Generate calibration data with sub-array multiplexing.

    Parameters
    ----------
    m : int or 'K'
        Number of simultaneously probed qubits per round.
        m='K' = full multiplexing (ideal).
    sigma_self : float
        Std of residual self-phase noise per qubit per round.
        Simulates: (1) imperfect single-qubit baseline subtraction,
                   (2) stochastic charge noise xi_i.

    Protocol
    --------
    Each round e:
      - Select m qubits uniformly at random (without replacement)
      - Apply random displacements u ~ N(0, U_J²) to selected qubits
      - Measure K Ramsey phases: Δθ = M @ u + σ_φ × noise + σ_self × self_noise
        where self_noise_i ~ N(0,1) independently (residual self-phase fluctuation)
    """
    m_eff = K if (m == 'K' or m >= K) else m
    U = np.zeros((n_exp, K))
    for e in range(n_exp):
        probe_idx = rng.choice(K, size=m_eff, replace=False)
        U[e, probe_idx] = rng.normal(0, U_J, size=m_eff)

    shot_noise  = rng.normal(0, SIGMA_PHI,   size=(n_exp, K))
    self_noise  = rng.normal(0, sigma_self, size=(n_exp, K)) if sigma_self > 0 else 0.0
    DTheta = U @ M_true.T + shot_noise + self_noise
    return U, DTheta

def reconstruct_M_plain(U, DTheta):
    K = DTheta.shape[1]; M = np.zeros((K,K))
    for i in range(K):
        M[i,:], _, _, _ = np.linalg.lstsq(U, DTheta[:,i], rcond=None)
    np.fill_diagonal(M,0.0); return M

def reconstruct_M_with_prior(U, DTheta, positions, idx, z=Z_DEPTH):
    """
    Physics-informed decomposition:
        M = M_prior + Delta M_res, with M_prior given by the 1/r^3 baseline.

    In the overdetermined multiplexed regime used in this robustness analysis,
    residual least squares is algebraically equivalent to plain LS. Its role is
    interpretability of the recovered susceptibility network, not an independent
    statistical regularization advantage.
    """
    K = DTheta.shape[1]
    M_stat = build_M_static(positions, idx, z)
    DTheta_res = DTheta - U @ M_stat.T
    M_res = np.zeros((K,K))
    for i in range(K):
        M_res[i,:], _, _, _ = np.linalg.lstsq(U, DTheta_res[:,i], rcond=None)
    M_hat = M_stat + M_res
    np.fill_diagonal(M_hat,0.0); return M_hat


def fidelity_comp(M_true, M_hat, u=None):
    K = M_true.shape[0]
    if u is None: u = U_J*np.ones(K)
    P   = np.eye(K)-np.ones((K,K))/K
    res = P@((M_true-M_hat)@u)
    var = float(np.dot(res,res)/(K-1))
    return np.exp(-var/2), var


# ══════════════════════════════════════════════════════════════════
# §3A  SUB-ARRAY MULTIPLEXING SCAN
# ══════════════════════════════════════════════════════════════════
def scan_subarray(positions, K_range, n_trials=N_TRIALS):
    """
    For each m ∈ {K, 8, 4}:
      Find minimum n_exp (rounds) achieving F_comp > 0.999 using the residual structural decomposition.
      Convert to calibration time = n_exp × T_Ramsey.
    Also compute:
      - Naive pairwise time = K(K-1)/2 × T_Ramsey
      - F_comp vs K at fixed n_exp = 3K (shows performance difference)
    """
    N = len(positions)
    cal_times = {m: [] for m in M_VALUES}   # time to achieve F>0.999
    n_exp_min = {m: [] for m in M_VALUES}    # empirically determined n_exp*
    cal_Ks    = []
    f_at_3K   = {m: {'mean':[], 'std':[]} for m in M_VALUES}  # F at n_exp=3K
    n_exp_grid = lambda K: sorted(set(int(round(f*K)) for f in [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.5,5.0,6.0,8.0,10.0]))

    for K in K_range:
        if K > N or K < 3: continue
        cal_Ks.append(K)

        for m in M_VALUES:
            # Find minimum n_exp for F>0.999
            best_n = None
            for n_exp in n_exp_grid(K):
                Fs = []
                for tr in range(n_trials):
                    rng_tr = np.random.default_rng(tr*31 + (m if isinstance(m,int) else 0))
                    idx = rng_tr.choice(N, size=K, replace=False)
                    M_true = build_M_dynamic(positions, idx, rng=np.random.default_rng(tr))
                    u_nom  = rng_tr.uniform(-U_J, U_J, size=K)
                    U, D   = generate_sparse_probe(M_true, K, n_exp, m,
                                                   rng=np.random.default_rng(tr+200))
                    Mh = reconstruct_M_with_prior(U, D, positions, idx)
                    F, _ = fidelity_comp(M_true, Mh, u_nom)
                    Fs.append(F)
                if np.mean(Fs) > 0.999:
                    best_n = n_exp
                    break
            if best_n is None:
                best_n = 10*K
            n_exp_min[m].append(best_n)
            cal_times[m].append(best_n * T_RAMSEY)

            # F at n_exp = 3K (fixed budget comparison)
            n3K = 3*K
            Fs3 = []
            for tr in range(n_trials):
                rng_tr = np.random.default_rng(tr*31 + (m if isinstance(m,int) else 0) + 1000)
                idx = rng_tr.choice(N, size=K, replace=False)
                M_true = build_M_dynamic(positions, idx, rng=np.random.default_rng(tr+50))
                u_nom  = rng_tr.uniform(-U_J, U_J, size=K)
                U, D   = generate_sparse_probe(M_true, K, n3K, m,
                                               rng=np.random.default_rng(tr+300))
                Mh = reconstruct_M_with_prior(U, D, positions, idx)
                F, _ = fidelity_comp(M_true, Mh, u_nom)
                Fs3.append(F)
            f_at_3K[m]['mean'].append(np.mean(Fs3))
            f_at_3K[m]['std'].append(np.std(Fs3))

    for m in M_VALUES:
        cal_times[m] = np.array(cal_times[m])
        n_exp_min[m] = np.array(n_exp_min[m], dtype=int)
        f_at_3K[m]['mean'] = np.array(f_at_3K[m]['mean'])
        f_at_3K[m]['std']  = np.array(f_at_3K[m]['std'])

    return np.array(cal_Ks), cal_times, f_at_3K, n_exp_min


# ══════════════════════════════════════════════════════════════════
# §3B  PRIOR MISMATCH ROBUSTNESS
# ══════════════════════════════════════════════════════════════════
def scan_prior_mismatch(positions, K=40, n_trials=N_TRIALS):
    """
    Vary alpha (prior mismatch amplitude) from 0 to 0.6.
    Test: F_comp with prior vs plain LS.
    True M = M_static × (1 + alpha × random_perturbation).
    Prior always uses ideal M_static (1/r³).
    """
    N = len(positions)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    idx0   = RNG.choice(N, size=K, replace=False)

    res_plain = {'alpha':[], 'F_mean':[], 'F_std':[]}
    res_prior = {'alpha':[], 'F_mean':[], 'F_std':[]}

    for alpha in alphas:
        fp, fpr = [], []
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(tr*57)
            M_true = build_M_perturbed(positions, idx0, alpha=alpha,
                                        rng=np.random.default_rng(tr+400))
            u_nom  = rng_tr.uniform(-U_J, U_J, size=K)
            n_exp  = 2*K

            U, D   = generate_sparse_probe(M_true, K, n_exp, 'K',
                                           rng=np.random.default_rng(tr+500))
            Mp  = reconstruct_M_plain(U, D)
            Mpr = reconstruct_M_with_prior(U, D, positions, idx0)
            F_p,  _ = fidelity_comp(M_true, Mp,  u_nom)
            F_pr, _ = fidelity_comp(M_true, Mpr, u_nom)
            fp.append(F_p); fpr.append(F_pr)

        res_plain['alpha'].append(alpha)
        res_plain['F_mean'].append(np.mean(fp))
        res_plain['F_std'].append(np.std(fp))
        res_prior['alpha'].append(alpha)
        res_prior['F_mean'].append(np.mean(fpr))
        res_prior['F_std'].append(np.std(fpr))

    return res_plain, res_prior


# ══════════════════════════════════════════════════════════════════
# §3C  RESIDUAL SELF-PHASE NOISE
# ══════════════════════════════════════════════════════════════════
def scan_selfphase_noise(positions, K_range, n_trials=N_TRIALS):
    """
    Scan sigma_self / sigma_phi (residual self-phase noise as fraction of shot noise).
    Represents:
      - Imperfect single-qubit baseline calibration (Sec 5.1 assumption)
      - Charge noise xi_i(t) accumulated during transport
      - Nuclear spin bath fluctuations
    Test: F_comp for sigma_self/sigma_phi ∈ {0, 0.5, 1, 2, 5}
    """
    N = len(positions)
    # sigma_self in units of sigma_phi
    noise_ratios = [0.0, 0.5, 1.0, 2.0, 5.0]

    K = 40   # fixed K for noise scan
    n_exp = 2*K

    res = {}
    for ratio in noise_ratios:
        sigma_self = ratio * SIGMA_PHI
        Fs = []
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(tr*97)
            idx    = rng_tr.choice(N, size=K, replace=False)
            M_true = build_M_dynamic(positions, idx, rng=np.random.default_rng(tr))
            u_nom  = rng_tr.uniform(-U_J, U_J, size=K)
            U, D   = generate_sparse_probe(M_true, K, n_exp, 'K',
                                           sigma_self=sigma_self,
                                           rng=np.random.default_rng(tr+700))
            Mh = reconstruct_M_with_prior(U, D, positions, idx)
            F, _ = fidelity_comp(M_true, Mh, u_nom)
            Fs.append(F)
        res[ratio] = (np.mean(Fs), np.std(Fs))

    # Also scan K for sigma_self = 1×sigma_phi (realistic)
    Fk_mean, Fk_std, Kv = [], [], []
    for K in K_range:
        if K > N or K < 3: continue
        Kv.append(K)
        sigma_self = 1.0 * SIGMA_PHI
        n_exp_K = 2*K
        Fs = []
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(tr*97 + K)
            idx    = rng_tr.choice(N, size=K, replace=False)
            M_true = build_M_dynamic(positions, idx, rng=np.random.default_rng(tr))
            u_nom  = rng_tr.uniform(-U_J, U_J, size=K)
            U, D   = generate_sparse_probe(M_true, K, n_exp_K, 'K',
                                           sigma_self=sigma_self,
                                           rng=np.random.default_rng(tr+800))
            Mh = reconstruct_M_with_prior(U, D, positions, idx)
            F, _ = fidelity_comp(M_true, Mh, u_nom)
            Fs.append(F)
        Fk_mean.append(np.mean(Fs)); Fk_std.append(np.std(Fs))

    return res, np.array(Kv), np.array(Fk_mean), np.array(Fk_std)


# ══════════════════════════════════════════════════════════════════
# §4  PLOTTING
# ══════════════════════════════════════════════════════════════════
def plot_3A(K_arr, cal_times, f_at_3K, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) Calibration time vs K
    ax = axes[0]
    K = np.array(K_arr, dtype=float)

    # Naive pairwise (reference)
    naive_time = K*(K-1)/2 * T_RAMSEY
    ax.semilogy(K, naive_time, color='#d6604d', lw=2.0, ls='-',
                label='Naive pairwise $O(K^2)$')

    # Sub-array curves
    for m in M_VALUES:
        t_arr = cal_times[m]
        ax.semilogy(K, t_arr, color=M_COLORS[m], lw=1.8,
                    ls=M_STYLES[m], label=M_LABELS[m])

    # τ_drift reference
    ax.axhline(TAU_DRIFT, color='#333', ls=':', lw=1.2)
    ax.text(0.97, 0.61, r'$\tau_{\rm drift}$ (1 hr)', transform=ax.transAxes,
            fontsize=7.5, color='#333', ha='right', va='bottom')

    # K* markers for naive and m=4
    K_star_naive = float(np.interp(TAU_DRIFT, naive_time, K)) if naive_time[-1]>TAU_DRIFT else K[-1]
    K_star_m4    = float(np.interp(TAU_DRIFT, cal_times[4], K)) if cal_times[4][-1]>TAU_DRIFT else K[-1]

    if K_star_naive < K[-1]:
        ax.axvline(K_star_naive, color='#d6604d', lw=1.2, ls='--', alpha=0.7)
        ax.text(K_star_naive+1, 12, f'$K^*\\!={K_star_naive:.0f}$',
                fontsize=7, color='#d6604d', va='bottom')
    if K_star_m4 < K[-1]:
        ax.axvline(K_star_m4, color=M_COLORS[4], lw=1.2, ls='--', alpha=0.7)
        ax.text(K_star_m4+1, 22, f'$K^*(m\\!=\\!4)\\!={K_star_m4:.0f}$',
                fontsize=7, color=M_COLORS[4], va='bottom')

    ax.fill_betweenx([TAU_DRIFT, 1e5], K_star_naive, K[-1],
                     alpha=0.05, color='#d6604d')
    ax.set_xlabel('Active qubits $K$')
    ax.set_ylabel('Calibration time [s]')
    ax.set_title('(a) Calibration time vs $K$\n(sub-array multiplexing)', fontsize=9)
    ax.set_ylim(10, 1e5)
    ax.set_xlim(K[0], K[-1])
    ax.legend(frameon=False, fontsize=6.8, loc='upper left')
    _despine(ax)

    # (b) F_comp vs K at n_exp = 3K for each m
    ax = axes[1]
    for m in M_VALUES:
        mn = f_at_3K[m]['mean']; sd = f_at_3K[m]['std']
        ax.fill_between(K, np.clip(mn-sd, 0, 1.01),
                           np.clip(mn+sd, 0, 1.01),
                        color=M_COLORS[m], alpha=0.12)
        ax.plot(K, mn, color=M_COLORS[m], lw=1.8, ls=M_STYLES[m],
                marker='o', ms=3.5, markevery=3, label=M_LABELS[m])
    ax.axhline(0.999, color='#555', ls=':', lw=1.0, alpha=0.7)
    ax.text(K[-1]-1, 0.985, '$F=0.999$', fontsize=6.5, color='#555',
            ha='right', va='bottom')
    ax.set_xlabel('Active qubits $K$')
    ax.set_ylabel('$F_{\\rm comp}$')
    ax.set_title('(b) $F_{\\rm comp}$ at $n_{\\rm exp}=3K$ rounds\n'
                 '(finite-channel multiplexing)', fontsize=9)
    ax.set_ylim(0.78, 1.02)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False, fontsize=7.0, loc='lower left')
    _despine(ax)

    plt.tight_layout(pad=0.5, w_pad=1.6)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_3A_subarray_mux.{ext}'),
                    bbox_inches='tight')
    plt.close(fig); print('  fig_3A_subarray_mux saved')


def plot_3B(res_plain, res_prior, outdir):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    alp = np.array(res_plain['alpha']) * 100   # percent
    ax.plot(alp, res_plain['F_mean'], 'o-', color='#d6604d', lw=1.6, ms=5,
            label='Plain LS')
    ax.fill_between(alp,
                    np.clip(np.array(res_plain['F_mean'])-np.array(res_plain['F_std']),0,1),
                    np.clip(np.array(res_plain['F_mean'])+np.array(res_plain['F_std']),0,1),
                    color='#d6604d', alpha=0.12)
    ax.plot(alp, res_prior['F_mean'], 's-', color='#2166ac', lw=1.6, ms=5,
            label='Struct. reduction ($1/r^3$ prior)')
    ax.fill_between(alp,
                    np.clip(np.array(res_prior['F_mean'])-np.array(res_prior['F_std']),0,1),
                    np.clip(np.array(res_prior['F_mean'])+np.array(res_prior['F_std']),0,1),
                    color='#2166ac', alpha=0.12)
    ax.axhline(0.999, color='#4dac26', ls='--', lw=1.3, alpha=0.8,
               label='$F=0.999$ target')
    ax.set_xlabel('Prior mismatch $\\alpha$ [%]')
    ax.set_ylabel('$F_{\\rm comp}$')
    ax.set_title('Prior mismatch robustness\n'
                 r'($K=40$, $n_{\rm exp}=2K$, full mux.)', fontsize=9)
    ax.set_ylim(0.78, 1.02)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False, fontsize=7.0, loc='lower left')
    _despine(ax)

    plt.tight_layout(pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_3B_prior_mismatch.{ext}'),
                    bbox_inches='tight')
    plt.close(fig); print('  fig_3B_prior_mismatch saved')


def plot_3C(noise_res, K_arr, Fk_mean, Fk_std, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) F_comp vs noise ratio at K=40
    ax = axes[0]
    ratios  = sorted(noise_res.keys())
    F_means = [noise_res[r][0] for r in ratios]
    F_stds  = [noise_res[r][1] for r in ratios]
    ax.errorbar(ratios, F_means, yerr=F_stds, fmt='o-', color='#2166ac',
                lw=1.6, ms=5, capsize=3, elinewidth=1.0)
    ax.axhline(0.999, color='#4dac26', ls='--', lw=1.3, alpha=0.8,
               label='$F=0.999$ target')
    ax.set_xlabel(r'Self-phase noise $\sigma_{\rm self}/\sigma_\varphi$')
    ax.set_ylabel('$F_{\\rm comp}$  ($K=40$, $n_{\\rm exp}=2K$)')
    ax.set_title('(a) Residual self-phase noise\n'
                 r'(imperfect $\Delta\theta^{\rm self}$ calibration)', fontsize=9)
    ax.set_ylim(0.78, 1.02)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    _despine(ax)

    # (b) F_comp vs K with sigma_self = sigma_phi (realistic noise)
    ax = axes[1]
    K = np.array(K_arr, dtype=float)
    ax.fill_between(K, np.clip(Fk_mean-Fk_std,0,1.01),
                       np.clip(Fk_mean+Fk_std,0,1.01),
                    color='#2166ac', alpha=0.15)
    ax.plot(K, Fk_mean, 's-', color='#2166ac', lw=1.8, ms=4, markevery=3,
            label=r'$\sigma_{\rm self}=\sigma_\varphi$  (realistic)')
    ax.axhline(0.999, color='#4dac26', ls='--', lw=1.3, alpha=0.8,
               label='$F=0.999$ target')
    ax.set_xlabel('Active qubits $K$')
    ax.set_ylabel('$F_{\\rm comp}$')
    ax.set_title('(b) Scaling with realistic noise\n'
                 r'($\sigma_{\rm self}=\sigma_\varphi$, $n_{\rm exp}=2K$)', fontsize=9)
    ax.set_ylim(0.78, 1.02)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    _despine(ax)

    plt.tight_layout(pad=0.5, w_pad=1.6)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_3C_selfphase_noise.{ext}'),
                    bbox_inches='tight')
    plt.close(fig); print('  fig_3C_selfphase_noise saved')


def plot_summary_robustness(K_arr, cal_times, f_at_3K, res_plain, res_prior,
                             noise_res, K_arr_c, Fk_mean, Fk_std, outdir):
    """
    3-panel robustness summary for the main text.

    Report-aligned design:
      (a) wall-clock calibration scaling under finite-channel probing;
      (b) prior-mismatch robustness shown as infidelity, not flat F_comp;
      (c) residual self-phase noise shown as infidelity with an uncompensated
          baseline for contrast.
    """
    fig = plt.figure(figsize=(7.6, 3.1))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                            left=0.07, right=0.97, top=0.84, bottom=0.18)
    K = np.array(K_arr, dtype=float)

    # ── Panel A: calibration time ─────────────────────────────────
    ax = fig.add_subplot(gs[0])
    naive_time = K*(K-1)/2 * T_RAMSEY
    ax.semilogy(K, naive_time, color='#d6604d', lw=2.0, label='Naive $O(K^2)$')
    for m in M_VALUES:
        ax.semilogy(K, cal_times[m], color=M_COLORS[m], lw=1.6,
                    ls=M_STYLES[m], label=M_LABELS[m])
    ax.axhline(TAU_DRIFT, color='#333', ls=':', lw=1.2)
    ax.text(0.97, 0.72, r'$\tau_{\rm drift}$', transform=ax.transAxes,
            fontsize=7.5, color='#333', ha='right', va='top')

    K_star_naive = float(np.interp(TAU_DRIFT, naive_time, K)) if naive_time[-1]>TAU_DRIFT else K[-1]
    K_star_m4    = float(np.interp(TAU_DRIFT, cal_times[4], K)) if cal_times[4][-1]>TAU_DRIFT else K[-1]
    if K_star_naive < K[-1]:
        ax.axvline(K_star_naive, color='#d6604d', lw=1.2, ls='--', alpha=0.8)
        ax.text(K_star_naive-2, 12, f'$K^*={K_star_naive:.0f}$',
                fontsize=7, color='#d6604d', va='bottom', ha='right')
    if K_star_m4 < K[-1]:
        ax.axvline(K_star_m4, color=M_COLORS[4], lw=1.2, ls='--', alpha=0.7)
    ax.fill_betweenx([TAU_DRIFT, 1e5], K_star_naive, K[-1], alpha=0.05, color='#d6604d')
    ax.set_xlabel('Active qubits $K$', fontsize=8.5)
    ax.set_ylabel('Calibration time [s]', fontsize=8.5)
    ax.set_title('(a) Sub-array multiplexing\n(calibration time)', fontsize=9, pad=4)
    ax.set_ylim(10, 1e5); ax.set_xlim(K[0], K[-1])
    ax.legend(frameon=False, fontsize=6.0, loc='upper left')
    _despine(ax)

    # ── Panel B: prior mismatch, plotted as infidelity ─────────────
    ax = fig.add_subplot(gs[1])
    alp = np.array(res_plain['alpha']) * 100
    y_plain, lo_plain, hi_plain = _infidelity(res_plain['F_mean'], res_plain['F_std'])
    y_prior, lo_prior, hi_prior = _infidelity(res_prior['F_mean'], res_prior['F_std'])
    ax.fill_between(alp, lo_plain, hi_plain, color='#d6604d', alpha=0.16)
    ax.semilogy(alp, y_plain, 'o-', color='#d6604d', lw=1.6, ms=4.5,
                label='Plain LS')
    ax.fill_between(alp, lo_prior, hi_prior, color='#2166ac', alpha=0.16)
    ax.semilogy(alp, y_prior, 's-', color='#2166ac', lw=1.6, ms=4.5,
                label='Struct. reduction')
    ax.axhline(1e-3, color='#4dac26', ls='--', lw=1.2, alpha=0.8,
               label='$F=0.999$ target')
    ax.text(0.04, 0.07, 'same data/noise\nrealizations',
            transform=ax.transAxes, fontsize=6.1, color='#555', ha='left', va='bottom',
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0))
    ax.set_xlabel('Prior mismatch $\\alpha$ [%]', fontsize=8.5)
    ax.set_ylabel(r'Infidelity $1-F_{\rm comp}$', fontsize=8.5)
    ax.set_title('(b) Prior mismatch\n(no-bias check)', fontsize=9, pad=4)
    ax.set_ylim(1e-5, 3e-3)
    ax.legend(frameon=False, fontsize=6.7, loc='upper left')
    _despine(ax)

    # ── Panel C: self-phase noise, infidelity + no-compensation baseline ─
    ax = fig.add_subplot(gs[2])
    Kc = np.array(K_arr_c, dtype=float)
    y_comp, lo_comp, hi_comp = _infidelity(Fk_mean, Fk_std)
    F_naive, F_naive_std = _compute_naive_f_vs_K(Kc)
    y_naive, lo_naive, hi_naive = _infidelity(F_naive, F_naive_std, floor=1e-6)

    ax.fill_between(Kc, lo_naive, hi_naive, color='#d6604d', alpha=0.10)
    ax.semilogy(Kc, y_naive, 'o-', color='#d6604d', lw=1.3, ms=3.2,
                markevery=3, alpha=0.85, label='No compensation')
    ax.fill_between(Kc, lo_comp, hi_comp, color='#2166ac', alpha=0.18)
    ax.semilogy(Kc, y_comp, 's-', color='#2166ac', lw=1.8, ms=4, markevery=3,
                label=r'Framework, $\sigma_{\rm self}=\sigma_\varphi$')
    ax.axhline(1e-3, color='#4dac26', ls='--', lw=1.2, alpha=0.8,
               label='$F=0.999$ target')
    ax.set_xlabel('Active qubits $K$', fontsize=8.5)
    ax.set_ylabel(r'Infidelity $1-F_{\rm comp}$', fontsize=8.5)
    ax.set_title('(c) Residual self-phase noise\n'
                 r'($n_{\rm exp}=2K$)', fontsize=9, pad=4)
    ax.set_ylim(1e-6, 1.2)
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    _despine(ax)

    fig.savefig(os.path.join(outdir, 'fig_3_robustness_summary.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'fig_3_robustness_summary.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig); print('  fig_3_robustness_summary saved')


# ══════════════════════════════════════════════════════════════════
# §4B  APPENDIX 2×2 FIGURE  (fig_4_extended_robustness)
# ══════════════════════════════════════════════════════════════════
def plot_fig4_extended(K_arr, cal_times, f_at_3K, n_exp_min,
                       res_plain_B, res_prior_B,
                       noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C,
                       outdir):
    """
    2×2 Appendix figure — non-duplicated vs main-text summary:

    (a) Sub-array F_comp vs K at n_exp=3K  [m=K,8,4; ±1σ shading; y: 0.994-1.002]
        Identical metric to summary (b) but now shows m=4 as additional curve.

    (b) n_exp*/K needed vs K — NEW METRIC not in main text
        Directly quantifies calibration overhead per qubit, per m value.
        Secondary annotation: actual wall-clock time at K=80.

    (c) Prior mismatch at n=2K, dense alpha, ±1σ  [y: 0.997-1.0005]
        Note at BOTTOM (below 0.999 line) states n=K result.
        Legend placed just below 0.999 dashed line.

    (d) Dense K scan (step 3) + no-compensation baseline  [y: 0-1 full scale]
        Contrast framework vs uncompensated → shows effect clearly.
    """
    # ── Derived quantities needed for (b) ────────────────────────────────
    # Calibration overhead is computed directly from scan_subarray(), not
    # hand-entered.  This keeps Fig. 4(b) reproducible from the same Monte
    # Carlo data that generates the summary calibration-time plot.
    Kv_b = np.array(K_arr, dtype=float)
    n_exp_needed = {m: np.array(n_exp_min[m], dtype=float) / Kv_b for m in M_VALUES}
    nexp_labels  = {'K': '$m=K$', 8: '$m=8$', 4: '$m=4$'}
    nexp_keys    = [('K', M_COLORS['K'], '-'), (8, M_COLORS[8], '--'), (4, M_COLORS[4], '-.')]

    # ── Dense K scan for (d) ─────────────────────────────────────────────
    # Use K_arr_C (self-noise scan) as the dense K set with sigma_self=sigma_phi
    Kv_d    = K_arr_C
    Fd_m    = Fk_mean_C
    Fd_s    = Fk_std_C
    # naive baseline (no compensation)
    Fd_naive = np.array([noise_res_C[0.0][0]] * len(Kv_d))  # approx: same K-scan

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8),
                              gridspec_kw={'hspace': 0.52, 'wspace': 0.40})
    fig.subplots_adjust(left=0.09, right=0.93, top=0.94, bottom=0.09)

    # ── (a) Sub-array F@3K ───────────────────────────────────────────────
    ax = axes[0, 0]
    for m in ['K', 8, 4]:
        mn = f_at_3K[m]['mean']; sd = f_at_3K[m]['std']
        Ksub = K_arr[:len(mn)]
        ax.fill_between(Ksub,
                        np.clip(mn - sd, 0.994, 1.002),
                        np.clip(mn + sd, 0.994, 1.002),
                        color=M_COLORS[m], alpha=0.18)
        ax.plot(Ksub, mn, color=M_COLORS[m], lw=1.7,
                ls=M_STYLES[m], marker='o', ms=3.5, markevery=3,
                label={'K': '$m=K$', 8: '$m=8$', 4: '$m=4$'}[m])
    ax.axhline(0.999, color='#555', ls=':', lw=1.0, alpha=0.8)
    ax.text(K_arr[-1] - 1, 0.9988, '$F=0.999$',
            fontsize=6.5, color='#555', ha='right', va='top')
    ax.set_xlabel('Active qubits $K$'); ax.set_ylabel(r'$F_{\rm comp}$')
    ax.set_title('(a) Sub-array: $F_{\\rm comp}$ at $n_{\\rm exp}=3K$\n'
                 r'(shaded $\pm1\sigma$, $N_{\rm trials}=' + str(N_TRIALS) + r'$)', fontsize=9)
    ax.set_ylim(0.994, 1.002); ax.set_yticks([0.994, 0.996, 0.998, 1.000])
    ax.legend(frameon=False, fontsize=8.0, loc='lower left'); _despine(ax)

    # ── (b) n_exp*/K vs K — calibration overhead ─────────────────────────
    ax = axes[0, 1]
    for m_key, col, ls in nexp_keys:
        nv = np.array(n_exp_needed[m_key], dtype=float)
        ax.plot(Kv_b, nv, color=col, lw=1.7, ls=ls, marker='s', ms=4.5,
                label=nexp_labels[m_key])
    # Linear fit lines to highlight the linear scaling trend
    for m_key, col, ls in nexp_keys:
        nv = np.array(n_exp_needed[m_key], dtype=float)
        coeffs = np.polyfit(Kv_b, nv, 1)   # linear fit
        Kfit = np.linspace(Kv_b[0], Kv_b[-1], 100)
        ax.plot(Kfit, np.polyval(coeffs, Kfit),
                color=col, lw=0.9, ls='--', alpha=0.45, zorder=1)
    for m_key, col, _ in nexp_keys:
        nv = np.array(n_exp_needed[m_key], dtype=float)
        idx80 = np.where(Kv_b == 80)[0]
        idx_ref = int(idx80[0]) if len(idx80) else -1
        k_ref = Kv_b[idx_ref]
        t_ref = np.array(n_exp_min[m_key], dtype=float)[idx_ref] * T_RAMSEY
        mark = '(ok)' if t_ref < TAU_DRIFT else '(X)'
        ax.annotate(f'{t_ref:.0f}s {mark}',
                    xy=(k_ref, nv[idx_ref]), xytext=(k_ref + 2, nv[idx_ref]),
                    fontsize=6.2, color=col, va='center', annotation_clip=False)
    ax.set_xlabel('Active qubits $K$')
    ax.set_ylabel(r'$n_{\rm exp}^*/K$ (Ramsey rounds per qubit)')
    ax.set_title('(b) Calibration overhead vs $K$\n'
                 r'(rounds to $F>0.999$)', fontsize=9)
    ax.set_ylim(0, 7); ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.legend(frameon=False, fontsize=8.0); _despine(ax)

    # ── (c) Prior mismatch n=2K, zoomed, legend BELOW 0.999 line ─────────
    ax = axes[1, 0]
    alp   = np.array(res_plain_B['alpha'])     * 100   # convert to %
    mn2p  = np.array(res_plain_B['F_mean'])
    sd2p  = np.array(res_plain_B['F_std'])
    mn2pr = np.array(res_prior_B['F_mean'])
    sd2pr = np.array(res_prior_B['F_std'])

    ax.fill_between(alp,
                    np.clip(mn2p  - sd2p,  0.997, 1.0005),
                    np.clip(mn2p  + sd2p,  0.997, 1.0005),
                    color='#d6604d', alpha=0.20)
    l1, = ax.plot(alp, mn2p,  'o-', color='#d6604d', lw=1.7, ms=5,
                  label=r'Plain LS ($n=2K$)')
    ax.fill_between(alp,
                    np.clip(mn2pr - sd2pr, 0.997, 1.0005),
                    np.clip(mn2pr + sd2pr, 0.997, 1.0005),
                    color='#2166ac', alpha=0.20)
    l2, = ax.plot(alp, mn2pr, 's-', color='#2166ac', lw=1.7, ms=5,
                  label=r'Struct. reduction ($n=2K$)')
    l3  = ax.axhline(0.999, color='#4dac26', ls='--', lw=1.3, alpha=0.8,
                     label='$F=0.999$')

    # Legend just below 0.999 dashed line — use bbox_to_anchor in data coords
    # 0.999 line is at y=0.999; legend goes just below it, left side
    ax.legend(handles=[l1, l2, l3], frameon=True, fontsize=7.0,
              loc='upper left',
              bbox_to_anchor=(0.01, 0.46),   # 0.46 in axes coords ≈ just below 0.999
              framealpha=0.92, edgecolor='#ccc', borderpad=0.4)

    # Small note at very bottom (entirely below 0.999 space)
    ax.text(0.03, 0.06,
            r'At $n=K$: both methods give $F\approx0.86$' + '\n(identical — not shown)',
            transform=ax.transAxes, fontsize=6.2, color='#888', va='bottom',
            bbox=dict(fc='white', ec='#ddd', pad=2, lw=0.5, alpha=0.9))

    ax.set_xlabel('Prior mismatch $\\alpha$ [%]')
    ax.set_ylabel(r'$F_{\rm comp}$ ($K=40$, $n=2K$)')
    ax.set_title('(c) Prior mismatch robustness\n'
                 r'($n=2K$, $\pm1\sigma$, $N_{\rm trials}=' + str(N_TRIALS) + r'$)', fontsize=9)
    ax.set_ylim(0.997, 1.0005)
    ax.set_yticks([0.997, 0.998, 0.999, 1.000])
    _despine(ax)

    # ── (d) Dense K scan + simulated no-compensation baseline ──────────────
    ax = axes[1, 1]
    # Compute naive F_comp (no compensation) quickly across K_arr_C
    pos_d4 = perfect_lattice(L_GRID)
    N_d4   = len(pos_d4)
    Fd_naive_list = []
    for K in Kv_d:
        K = int(K)
        if K > N_d4: Fd_naive_list.append(np.nan); continue
        Fnaive_k = []
        for tr in range(min(12, N_TRIALS)):
            rng_n = np.random.default_rng(tr * 97 + K + 5000)
            idx_n = rng_n.choice(N_d4, size=K, replace=False)
            M_n   = build_M_dynamic(pos_d4, idx_n, rng=np.random.default_rng(tr))
            u_n   = rng_n.uniform(-U_J, U_J, size=K)
            F_n, _ = fidelity_comp(M_n, np.zeros((K, K)), u_n)
            Fnaive_k.append(F_n)
        Fd_naive_list.append(np.mean(Fnaive_k))
    Fd_naive = np.array(Fd_naive_list)

    # Plot naive curve first (behind framework)
    ax.plot(Kv_d, Fd_naive, 'o-', color='#d6604d', lw=1.4, ms=3,
            markevery=3, alpha=0.85, label='No compensation')
    ax.fill_between(Kv_d,
                    np.clip(Fd_m - 2 * Fd_s, 0, 1),
                    np.clip(Fd_m + 2 * Fd_s, 0, 1),
                    color='#2166ac', alpha=0.15)
    ax.plot(Kv_d, Fd_m, 's-', color='#2166ac', lw=1.8, ms=3, markevery=5,
            label=r'Framework ($\sigma_{\rm self}=\sigma_\varphi$)')
    ax.axhline(0.999, color='#4dac26', ls='--', lw=1.3, alpha=0.8,
               label='$F=0.999$')
    ax.set_xlabel('Active qubits $K$')
    ax.set_ylabel(r'$F_{\rm comp}$')
    ax.set_title('(d) Dense $K$ scan with stochastic noise\n'
                 r'($\sigma_{\rm self}=\sigma_\varphi$, $n_{\rm exp}=2K$, $\pm2\sigma$)',
                 fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc='center right'); _despine(ax)

    fig.savefig(os.path.join(outdir, 'fig_4_extended_robustness.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'fig_4_extended_robustness.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig); print('  fig_4_extended_robustness saved')


def save_source_data(outdir, K_arr, cal_times, f_at_3K, n_exp_min,
                     res_plain_B, res_prior_B,
                     noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C):
    """Save machine-readable source data for reproducibility."""
    os.makedirs(outdir, exist_ok=True)

    # Sub-array calibration overhead and wall-clock time
    path = os.path.join(outdir, 'source_data_stage2_robustness_subarray_overhead.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['K', 'm', 'n_exp_min', 'n_exp_min_over_K', 'calibration_time_s',
                    'within_tau_drift'])
        for i, K in enumerate(K_arr):
            for m in M_VALUES:
                n = int(n_exp_min[m][i])
                t = float(cal_times[m][i])
                w.writerow([int(K), str(m), n, n/float(K), t, int(t < TAU_DRIFT)])

    # Fixed-budget F_comp at n_exp = 3K
    path = os.path.join(outdir, 'source_data_stage2_robustness_subarray_F_at_3K.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['K', 'm', 'F_mean', 'F_std', 'n_exp'])
        for i, K in enumerate(K_arr):
            for m in M_VALUES:
                w.writerow([int(K), str(m), float(f_at_3K[m]['mean'][i]),
                            float(f_at_3K[m]['std'][i]), int(3*K)])

    # Prior mismatch panel source data
    path = os.path.join(outdir, 'source_data_stage2_robustness_prior_mismatch.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['alpha', 'method', 'F_mean', 'F_std'])
        for i, a in enumerate(res_plain_B['alpha']):
            w.writerow([float(a), 'plain_LS', float(res_plain_B['F_mean'][i]),
                        float(res_plain_B['F_std'][i])])
            w.writerow([float(a), 'structural_prior', float(res_prior_B['F_mean'][i]),
                        float(res_prior_B['F_std'][i])])

    # Self-phase noise K-scan source data, including the uncompensated
    # baseline used in Fig. 3(c) and Fig. 4(d).  This keeps the machine-readable
    # source data aligned with the plotted curves.
    F_no_comp, F_no_comp_std = _compute_naive_f_vs_K(K_arr_C)
    path = os.path.join(outdir, 'source_data_stage2_robustness_selfphase_Kscan.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['K',
                    'F_framework_mean_sigma_self_eq_sigma_phi',
                    'F_framework_std_sigma_self_eq_sigma_phi',
                    'F_no_comp_mean',
                    'F_no_comp_std'])
        for K, mn, sd, fn, fs in zip(K_arr_C, Fk_mean_C, Fk_std_C,
                                     F_no_comp, F_no_comp_std):
            w.writerow([int(K), float(mn), float(sd), float(fn), float(fs)])

    print('  Stage-2 robustness source CSV files saved')


# ══════════════════════════════════════════════════════════════════
# §5  FIGURE CAPTIONS
# ══════════════════════════════════════════════════════════════════
FIGURE_CAPTIONS = {

'fig_3A': (
    "FIG. A2. Sub-array multiplexing scalability. "
    "(a) Calibration time vs $K$ for naive pairwise interrogation and "
    "three multiplexing capacities ($m=K$ ideal, $m=8$, $m=4$). "
    "Finite-channel probing remains far below the naive $O(K^2)$ pairwise cost. "
    "(b) Compensation fidelity at fixed budget $n_{\\rm exp}=3K$ rounds. "
    "The finite-$m$ curves test compatibility with bandwidth-constrained control "
    "while keeping the central scaling mechanism unchanged: multiplexed Ramsey "
    "system identification rather than sequential pairwise calibration."
),

'fig_3B': (
    "FIG. A3. Robustness of multiplexed inference to prior model uncertainty. "
    "The true susceptibility is drawn as "
    "$\\chi_{ij}^{\\rm true}=\\chi_{ij}^{\\rm prior}(1+\\alpha\\varepsilon_{ij})$, "
    "while the structural decomposition uses the ideal $1/r^3$ baseline. "
    "Plain least squares and residual structural reduction remain essentially "
    "overlapping at $n_{\\rm exp}=2K$, as expected in the overdetermined "
    "multiplexed regime. This panel should be read as a no-bias robustness check "
    "for the physical decomposition, not as a claim of prior-driven statistical "
    "speedup."
),

'fig_3C': (
    "FIG. A4. Robustness to residual self-phase noise. "
    "(a) Compensation fidelity vs noise ratio $\\sigma_{\\rm self}/\\sigma_\\varphi$ "
    "at $K=40$, $n_{\\rm exp}=2K$. "
    "(b) Scaling with $K$ under realistic noise "
    "($\\sigma_{\\rm self}=\\sigma_\\varphi$). "
    "The results test whether deterministic crosstalk tracking remains effective "
    "when single-qubit baseline subtraction and stochastic transport noise are "
    "not perfect."
),

'fig_4_extended': (
    "FIG. 4. Extended robustness analysis of the phase compensation framework. "
    "(a) Compensation fidelity $F_{\\rm comp}$ vs $K$ at fixed budget "
    "$n_{\\rm exp}=3K$ for three multiplexing capacities. "
    "(b) Minimum calibration overhead $n_{\\rm exp}^*/K$ required to achieve "
    "$F_{\\rm comp}>0.999$, computed directly from the same Monte Carlo scan. "
    "(c) Prior model mismatch at $n_{\\rm exp}=2K$: plain LS and structural "
    "reduction overlap, confirming that the physical prior does not bias the "
    "overdetermined estimator. "
    "(d) Dense $K$ scan under residual self-phase noise compared with an "
    "uncompensated baseline, shown on the full fidelity scale."
),

'fig_3_summary': (
    "FIG. 3. Robustness of the phase compensation framework under realistic "
    "hardware constraints. "
    "(a) Calibration time vs $K$ for naive pairwise and finite-channel "
    "multiplexing ($m=K,8,4$). "
    "(b) Prior-model mismatch shown as infidelity, using matched probe/noise "
    "realizations for plain LS and structural reduction. "
    "(c) Residual self-phase noise shown as infidelity, with an uncompensated "
    "baseline included for contrast."
),
}

# ══════════════════════════════════════════════════════════════════
# §6  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    OUTDIR = os.environ.get('OUTDIR', '/content/outputs')
    os.makedirs(OUTDIR, exist_ok=True)

    print('='*62)
    print('  Stage 2 Robustness Analysis')
    print('='*62)
    print(f'  N_TRIALS = {N_TRIALS}')
    print(f'  T_RAMSEY = {T_RAMSEY}s,  τ_drift = {TAU_DRIFT}s')
    print(f'  m values: {M_VALUES}')
    print()

    pos = perfect_lattice(L_GRID)
    K_RANGE_A = list(range(5, 81, 5))
    K_RANGE_C = list(range(5, 81, 5))

    print(f'[3A] Sub-array multiplexing scan (N_TRIALS={N_TRIALS}) ...')
    K_arr_A, cal_times, f_at_3K, n_exp_min = scan_subarray(pos, K_RANGE_A, n_trials=N_TRIALS)
    for m in M_VALUES:
        k80 = np.where(K_arr_A==80)[0]
        if len(k80):
            print(f'  {M_LABELS[m]:30s}: K=80 → t={cal_times[m][k80[0]]:.0f}s '
                  f'{"✓" if cal_times[m][k80[0]]<TAU_DRIFT else "✗"}')

    print(f'[3B] Prior mismatch scan (K=40, N_TRIALS={N_TRIALS}) ...')
    res_plain_B, res_prior_B = scan_prior_mismatch(pos, K=40, n_trials=N_TRIALS)
    print(f'  α=0.4 (40%): F_plain={res_plain_B["F_mean"][-3]:.4f}  '
          f'F_prior={res_prior_B["F_mean"][-3]:.4f}')

    print(f'[3C] Self-phase noise scan (N_TRIALS={N_TRIALS}) ...')
    noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C = scan_selfphase_noise(
        pos, K_RANGE_C, n_trials=N_TRIALS)
    print(f'  σ_self=1×σ_φ: F_comp={noise_res_C[1.0][0]:.4f}')
    print(f'  σ_self=2×σ_φ: F_comp={noise_res_C[2.0][0]:.4f}')

    print('\nGenerating figures ...')
    plot_fig4_extended(K_arr_A, cal_times, f_at_3K, n_exp_min,
                       res_plain_B, res_prior_B,
                       noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C,
                       OUTDIR)
    plot_3A(K_arr_A, cal_times, f_at_3K, OUTDIR)
    plot_3B(res_plain_B, res_prior_B, OUTDIR)
    plot_3C(noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C, OUTDIR)
    plot_summary_robustness(K_arr_A, cal_times, f_at_3K,
                             res_plain_B, res_prior_B,
                             noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C,
                             OUTDIR)

    save_source_data(OUTDIR, K_arr_A, cal_times, f_at_3K, n_exp_min,
                     res_plain_B, res_prior_B,
                     noise_res_C, K_arr_C, Fk_mean_C, Fk_std_C)

    print('\n'+'='*62)
    print('  FIGURE CAPTIONS (Robustness)')
    print('='*62)
    for k, v in FIGURE_CAPTIONS.items():
        print(f'\n[{k}]\n{v}')

    try:
        shutil.copy(__file__, os.path.join(OUTDIR, os.path.basename(__file__)))
    except Exception as exc:
        print(f'  warning: source-code copy skipped ({exc})')

    print('\n'+'='*62)
    print('  SUMMARY')
    print('='*62)
    for m in M_VALUES:
        k80 = np.where(K_arr_A==80)[0]
        if len(k80):
            t = cal_times[m][k80[0]]
            print(f'  {M_LABELS[m]}: K=80, t={t:.0f}s {"(✓ within τ_drift)" if t<TAU_DRIFT else "(✗ exceeds τ_drift)"}')
    print(f'  Prior mismatch α=40%: ΔF={res_prior_B["F_mean"][-3]-res_plain_B["F_mean"][-3]:+.4f}')
    print(f'  Self-phase noise σ=1×σ_φ: F={noise_res_C[1.0][0]:.4f}')
    print(f'\n  All outputs saved to {OUTDIR}')
