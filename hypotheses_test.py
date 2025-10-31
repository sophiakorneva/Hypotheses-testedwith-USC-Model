# --- Cell 1: imports & data (0.5 Gy) ---
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# экспериментальные точки (0.5 Gy)
t_exp = np.array([0.5, 2.0, 4.0, 6.0, 24.0], float)

data_gamma = {
    "gh2ax":     np.array([11.9, 8.85, 7.01, 6.06, 2.65], float),
    "gh2ax_err": np.array([0.31, 0.56, 0.51, 0.48, 0.14], float),
    "patm":      np.array([6.91, 4.65, 3.56, 2.83, 0.93], float),
    "patm_err":  np.array([0.39, 0.56, 0.43, 0.53, 0.25], float),
}
data_neutron = {
    "gh2ax":     np.array([10.63, 9.51, 8.86, 7.57, 6.00], float),
    "gh2ax_err": np.array([0.80, 0.37, 0.63, 0.47, 0.53], float),
    "patm":      np.array([6.49, 4.72, 3.95, 3.53, 2.41], float),
    "patm_err":  np.array([0.33, 0.64, 0.82, 0.51, 0.68], float),
}

DOSE_Gy = 0.5
LET_GAMMA, LET_NEUTRON = 0.2, 60.0

# best-fit C+ (из последнего прогона — фиксируем, чтобы варьировать только k_rs/w_dc при тестах)
P_BASE = {
    'k_rs':1.30524,'V_rc':1.37609,'K_rc':0.65230,'V_af':140.33535,'K_m':0.87532,
    'k_df':6.30090,'k_as':0.57916,'k_ds':0.09342,'k_p':10.63274,'k_g':5.15634,'w_dc':1.06114,
    'k_u':0.02174,'k_cu':0.03306,'s_du':0.34430
}
TH = {'LET50':8.42317,'f_min':0.44097,'f_max':0.55377}
PI_G, PI_N = 0.06974, 0.19862

# --- Cell 2: C+ model (rhs & simulate) ---

def split_damage(dose_Gy, LET, theta):
    f_min, f_max, LET50 = theta['f_min'], theta['f_max'], theta['LET50']
    f_c = f_min + (f_max - f_min) * (LET / (LET + LET50 + 1e-9))
    f_c = np.clip(f_c, 1e-6, 1.0 - 1e-6)
    D_tot = dose_Gy
    return (1.0 - f_c) * D_tot, f_c * D_tot, f_c

# states: [D_s, D_c, D_u, pATM_f, pATM_s, gH2AX]
def rhs_Cplus(t, y, p):
    D_s, D_c, D_u, pATM_f, pATM_s, gH2AX = y
    dD_s = -p['k_rs'] * D_s
    dD_c = -(p['V_rc'] * D_c) / (p['K_rc'] + D_c + 1e-9) - p['k_cu'] * D_c
    dD_u = -p['k_u'] * D_u + p['k_cu'] * D_c
    w_du = p['s_du'] * p['w_dc']
    weighted = D_s + p['w_dc'] * D_c + w_du * D_u
    prod_f = (p['V_af'] * weighted) / (p['K_m'] + weighted + 1e-9)
    dpATM_f = prod_f - p['k_df'] * pATM_f
    dpATM_s = p['k_as'] * pATM_f - p['k_ds'] * pATM_s
    dgH2AX  = p['k_p'] * (pATM_f + pATM_s) - p['k_g'] * gH2AX
    return [dD_s, dD_c, dD_u, dpATM_f, dpATM_s, dgH2AX]

def simulate_Cplus(params_gamma, params_neutron, LET, dose_Gy, t_eval, theta, pi_g, pi_n):
    D_s0, D_c0, _ = split_damage(dose_Gy, LET, theta)
    pi = pi_g if LET == LET_GAMMA else pi_n
    D_u0 = pi * D_c0
    D_c0 = (1.0 - pi) * D_c0
    y0 = [D_s0, D_c0, D_u0, 0.0, 0.0, 0.0]
    p = params_gamma if LET == LET_GAMMA else params_neutron
    sol = solve_ivp(lambda t,y: rhs_Cplus(t,y,p),
                    (0.0, float(np.max(t_eval))), y0, t_eval=np.asarray(t_eval,float),
                    method='LSODA', rtol=1e-6, atol=1e-8)
    # возвращаем gH2AX и pATM_f+pATM_s
    return sol.y[5], sol.y[3] + sol.y[4]

# --- Cell 3: helpers (norm, AIC, early slope) ---

@dataclass
class Series:
    t: np.ndarray; y: np.ndarray; e: np.ndarray|None = None

def _ensure_1d(a): return np.asarray(a, float).ravel()
def _clip_pos(a, eps=1e-12): return np.maximum(_ensure_1d(a), eps)

def norm_by_first(y, e=None):
    s = float(max(y[0], 1e-12))
    yn = y / s
    en = (e / s) if e is not None else None
    return yn, en, s

def aic_of(residuals: np.ndarray, k_params: int) -> float:
    r = _ensure_1d(residuals); n = r.size; rss = float(r @ r)
    return 2.0*k_params + n*np.log(max(rss/max(n,1), 1e-18))

def compute_keff(series: Series):
    t = _ensure_1d(series.t); z = np.log(_clip_pos(series.y))
    A = np.column_stack([np.ones_like(t), -t])
    theta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(theta[1])  # k_eff

# --- Cell 4: adapter & early-window slicing ---

class ModelAdapter:
    def __init__(self, P_base, TH, PI_g, PI_n):
        self.Pg = deepcopy(P_base)
        self.Pn = deepcopy(P_base)
        self.TH = deepcopy(TH); self.PI_g = PI_g; self.PI_n = PI_n
    def simulate_gh(self, radiation: str, t_grid: np.ndarray) -> np.ndarray:
        if radiation == "gamma":
            gh, _ = simulate_Cplus(self.Pg, self.Pn, LET_GAMMA, DOSE_Gy, t_grid, self.TH, self.PI_g, self.PI_n)
        elif radiation == "neutron":
            gh, _ = simulate_Cplus(self.Pg, self.Pn, LET_NEUTRON, DOSE_Gy, t_grid, self.TH, self.PI_g, self.PI_n)
        else:
            raise ValueError
        return gh

# раннее окно: возьмём все экспериментальные точки <= 2 ч
def early_window(data_dict):
    mask = t_exp <= 2.0
    return Series(t=t_exp[mask], y=data_dict["gh2ax"][mask], e=data_dict["gh2ax_err"][mask])

G_early = early_window(data_gamma)
N_early = early_window(data_neutron)

print("Раннее окно (ч):", G_early.t)

# --- Cell 5: residual builders & model comparison for H1 ---

# M1: k_rs split (k_rs^g, k_rs^n), MM и прочие параметры общие
def residuals_M1(theta, model: ModelAdapter, G: Series, N: Series):
    k_g, k_n = float(theta[0]), float(theta[1])
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_g; Pn['k_rs'] = k_n
    ygh = simulate_Cplus(Pg, Pn, LET_GAMMA, DOSE_Gy, G.t, model.TH, model.PI_g, model.PI_n)[0]
    ynh = simulate_Cplus(Pg, Pn, LET_NEUTRON, DOSE_Gy, N.t, model.TH, model.PI_g, model.PI_n)[0]
    ygh_n, eg, _ = norm_by_first(ygh, G.e); yg_n, _, _ = norm_by_first(G.y, G.e)
    ynh_n, en, _ = norm_by_first(ynh, N.e); yn_n, _, _ = norm_by_first(N.y, N.e)
    rg = (ygh_n - yg_n) / (eg if eg is not None else 1.0)
    rn = (ynh_n - yn_n) / (en if en is not None else 1.0)
    return np.r_[rg, rn]

# M2: k_rs shared, MM split по нейтронам (минимум — w_dc), это понадобится для H2-сравнения/контроля
MM_KEYS = ("w_dc",)  # можно расширить до ("V_rc","K_rc","w_dc","k_p","k_g")
def residuals_M2(theta, model: ModelAdapter, G: Series, N: Series):
    k_shared = float(theta[0]); deltas = theta[1:]
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_shared; Pn['k_rs'] = k_shared
    for i, key in enumerate(MM_KEYS):
        if key in Pn: Pn[key] = Pg[key] + deltas[i]
    ygh = simulate_Cplus(Pg, Pn, LET_GAMMA, DOSE_Gy, G.t, model.TH, model.PI_g, model.PI_n)[0]
    ynh = simulate_Cplus(Pg, Pn, LET_NEUTRON, DOSE_Gy, N.t, model.TH, model.PI_g, model.PI_n)[0]
    ygh_n, eg, _ = norm_by_first(ygh, G.e); yg_n, _, _ = norm_by_first(G.y, G.e)
    ynh_n, en, _ = norm_by_first(ynh, N.e); yn_n, _, _ = norm_by_first(N.y, N.e)
    return np.r_[ (ygh_n - yg_n)/(eg if eg is not None else 1.0),
                  (ynh_n - yn_n)/(en if en is not None else 1.0) ]

# M3: всё раздельно (диагностический «потолок» свободы)
def residuals_M3(theta, model: ModelAdapter, G: Series, N: Series):
    k_g, k_n = float(theta[0]), float(theta[1]); deltas = theta[2:]
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_g; Pn['k_rs'] = k_n
    for i, key in enumerate(MM_KEYS):
        if key in Pn: Pn[key] = Pg[key] + deltas[i]
    ygh = simulate_Cplus(Pg, Pn, LET_GAMMA, DOSE_Gy, G.t, model.TH, model.PI_g, model.PI_n)[0]
    ynh = simulate_Cplus(Pg, Pn, LET_NEUTRON, DOSE_Gy, N.t, model.TH, model.PI_g, model.PI_n)[0]
    ygh_n, eg, _ = norm_by_first(ygh, G.e); yg_n, _, _ = norm_by_first(G.y, G.e)
    ynh_n, en, _ = norm_by_first(ynh, N.e); yn_n, _, _ = norm_by_first(N.y, N.e)
    return np.r_[ (ygh_n - yg_n)/(eg if eg is not None else 1.0),
                  (ynh_n - yn_n)/(en if en is not None else 1.0) ]

# запуск сравнения
model = ModelAdapter(P_BASE, TH, PI_G, PI_N)

# sanity-check: лог-наклоны в раннем окне
kG = compute_keff(G_early); kN = compute_keff(N_early)
print(f"[Early 0.5–2h] k_eff^γ={kG:.3f}; k_eff^n={kN:.3f}; Δ={kN-kG:+.3f}")

# M1
x0_M1 = np.array([P_BASE['k_rs'], P_BASE['k_rs']*1.1])
res1 = least_squares(lambda th: residuals_M1(th, model, G_early, N_early),
                     x0=x0_M1, bounds=(np.array([1e-5,1e-5]), np.array([50.,50.])),
                     jac="2-point", method="trf", max_nfev=4000)
AIC1 = aic_of(res1.fun, k_params=2)

# M2 (минимальный сплит по w_dc)
x0_M2 = np.array([P_BASE['k_rs'], 0.0])   # [k_shared, delta_wdc]
lb_M2 = np.array([1e-5, -5.0]);  ub_M2 = np.array([50.0, 5.0])
res2 = least_squares(lambda th: residuals_M2(th, model, G_early, N_early),
                     x0=x0_M2, bounds=(lb_M2, ub_M2),
                     jac="2-point", method="trf", max_nfev=4000)
AIC2 = aic_of(res2.fun, k_params=1+len(MM_KEYS))

# M3
x0_M3 = np.array([P_BASE['k_rs'], P_BASE['k_rs']*1.1, 0.0])  # добавлен delta_wdc
lb_M3 = np.array([1e-5, 1e-5, -5.0]);  ub_M3 = np.array([50.0, 50.0, 5.0])
res3 = least_squares(lambda th: residuals_M3(th, model, G_early, N_early),
                     x0=x0_M3, bounds=(lb_M3, ub_M3),
                     jac="2-point", method="trf", max_nfev=5000)
AIC3 = aic_of(res3.fun, k_params=2+len(MM_KEYS))

best = min(AIC1, AIC2, AIC3)
print("\n=== MODEL COMPARISON (0.5–2 h) ===")
print(f"M1 (k_rs split, MM shared): AIC={AIC1:.2f}, k_rs^γ={res1.x[0]:.3g}, k_rs^n={res1.x[1]:.3g}, Δ={res1.x[1]-res1.x[0]:+.3g}")
print(f"M2 (k_rs shared, w_dc split): AIC={AIC2:.2f}, k_rs={res2.x[0]:.3g}, Δw_dc={res2.x[1]:+.3g}")
print(f"M3 (all split): AIC={AIC3:.2f}")
print(f"ΔAIC(M1–best)={AIC1-best:+.2f}; ΔAIC(M2–best)={AIC2-best:+.2f}; ΔAIC(M3–best)={AIC3-best:+.2f}")

# --- Add pATM into the early-window H1 test ---

# 1) соберём ранние ряды pATM (0.5–2 ч)
def early_window_patm(data_dict):
    mask = t_exp <= 2.0
    return Series(t=t_exp[mask], y=data_dict["patm"][mask], e=data_dict["patm_err"][mask])

Gp_early = early_window_patm(data_gamma)
Np_early = early_window_patm(data_neutron)

# 2) хелпер: стекуем остатки по gh2ax и pATM (нормируем каждый канал отдельно на свою 0.5 ч)
def _stack_resids(y_pred, y_obs, e_obs):
    y0, e0, _ = norm_by_first(y_obs, e_obs)
    yp, _, _ = norm_by_first(y_pred, None)
    return (yp - y0) / (e0 if e0 is not None else 1.0)

def residuals_M1_bivar(theta, model: ModelAdapter):
    k_g, k_n = float(theta[0]), float(theta[1])
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_g; Pn['k_rs'] = k_n
    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, G_early.t, model.TH, model.PI_g, model.PI_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, N_early.t, model.TH, model.PI_g, model.PI_n)
    r_gh = np.r_[_stack_resids(gh_g, G_early.y, G_early.e),
                 _stack_resids(gh_n, N_early.y, N_early.e)]
    r_pa = np.r_[_stack_resids(pa_g, Gp_early.y, Gp_early.e),
                 _stack_resids(pa_n, Np_early.y, Np_early.e)]
    return np.r_[r_gh, r_pa]

def residuals_M2_bivar(theta, model: ModelAdapter):
    k_shared = float(theta[0]); delta_w = float(theta[1])
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_shared; Pn['k_rs'] = k_shared
    Pn['w_dc'] = Pg['w_dc'] + delta_w
    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, G_early.t, model.TH, model.PI_g, model.PI_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, N_early.t, model.TH, model.PI_g, model.PI_n)
    r_gh = np.r_[_stack_resids(gh_g, G_early.y, G_early.e),
                 _stack_resids(gh_n, N_early.y, N_early.e)]
    r_pa = np.r_[_stack_resids(pa_g, Gp_early.y, Gp_early.e),
                 _stack_resids(pa_n, Np_early.y, Np_early.e)]
    return np.r_[r_gh, r_pa]

def residuals_M3_bivar(theta, model: ModelAdapter):
    k_g, k_n = float(theta[0]), float(theta[1]); delta_w = float(theta[2])
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_g; Pn['k_rs'] = k_n
    Pn['w_dc'] = Pg['w_dc'] + delta_w
    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, G_early.t, model.TH, model.PI_g, model.PI_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, N_early.t, model.TH, model.PI_g, model.PI_n)
    r_gh = np.r_[_stack_resids(gh_g, G_early.y, G_early.e),
                 _stack_resids(gh_n, N_early.y, N_early.e)]
    r_pa = np.r_[_stack_resids(pa_g, Gp_early.y, Gp_early.e),
                 _stack_resids(pa_n, Np_early.y, Np_early.e)]
    return np.r_[r_gh, r_pa]

# 3) прогоним сравнение M1/M2/M3 с бивариантными остатками
x0_M1b = np.array([P_BASE['k_rs'], P_BASE['k_rs']*1.1])
res1b = least_squares(lambda th: residuals_M1_bivar(th, model),
                      x0=x0_M1b, bounds=(np.array([1e-5,1e-5]), np.array([50.,50.])),
                      jac="2-point", method="trf", max_nfev=5000)
AIC1b = aic_of(res1b.fun, k_params=2)

x0_M2b = np.array([P_BASE['k_rs'], 0.0])
res2b = least_squares(lambda th: residuals_M2_bivar(th, model),
                      x0=x0_M2b, bounds=(np.array([1e-5,-5.0]), np.array([50.,5.0])),
                      jac="2-point", method="trf", max_nfev=5000)
AIC2b = aic_of(res2b.fun, k_params=2)

x0_M3b = np.array([P_BASE['k_rs'], P_BASE['k_rs']*1.1, 0.0])
res3b = least_squares(lambda th: residuals_M3_bivar(th, model),
                      x0=x0_M3b, bounds=(np.array([1e-5,1e-5,-5.0]), np.array([50.,50.,5.0])),
                      jac="2-point", method="trf", max_nfev=6000)
AIC3b = aic_of(res3b.fun, k_params=3)

bestb = min(AIC1b, AIC2b, AIC3b)
print("\n=== MODEL COMPARISON incl. pATM (0.5–2 h) ===")
print(f"M1b (k_rs split) : AIC={AIC1b:.2f}, k_rs^γ={res1b.x[0]:.3g}, k_rs^n={res1b.x[1]:.3g}, Δ={res1b.x[1]-res1b.x[0]:+.3g}")
print(f"M2b (w_dc split) : AIC={AIC2b:.2f}, k_rs={res2b.x[0]:.3g}, Δw_dc={res2b.x[1]:+.3g}")
print(f"M3b (all split)  : AIC={AIC3b:.2f}")
print(f"ΔAIC(M1b–best)={AIC1b-bestb:+.2f}; ΔAIC(M2b–best)={AIC2b-bestb:+.2f}; ΔAIC(M3b–best)={AIC3b-bestb:+.2f}")


# --- Cell 6: quick plot of early window (optional) ---
import matplotlib.pyplot as plt

Pg = deepcopy(model.Pg); Pn = deepcopy(model.Pn)
Pg['k_rs'] = float(res1.x[0]); Pn['k_rs'] = float(res1.x[1])
gh_g = simulate_Cplus(Pg, Pn, LET_GAMMA, DOSE_Gy, G_early.t, TH, PI_G, PI_N)[0]
gh_n = simulate_Cplus(Pg, Pn, LET_NEUTRON, DOSE_Gy, N_early.t, TH, PI_G, PI_N)[0]

yg0, _, _ = norm_by_first(G_early.y); ygh0, _, _ = norm_by_first(gh_g)
yn0, _, _ = norm_by_first(N_early.y); ynh0, _, _ = norm_by_first(gh_n)

plt.figure(figsize=(6,4))
plt.errorbar(G_early.t, yg0, yerr=(G_early.e/ max(G_early.y[0],1e-9)), fmt='o', label='γ data (norm)')
plt.plot(G_early.t, ygh0, '-s', label='γ model (M1)')
plt.errorbar(N_early.t, yn0, yerr=(N_early.e/ max(N_early.y[0],1e-9)), fmt='o', label='n data (norm)')
plt.plot(N_early.t, ynh0, '-s', label='n model (M1)')
plt.xlabel('Time (h)'); plt.ylabel('γH2AX (norm to 0.5 h)'); plt.title('Early window fit (0.5–2 h)')
plt.grid(True, alpha=0.3); plt.legend(); plt.show()

# Раннее окно (H1/H2-ранний тест)
mask_early = (t_exp <= 2.0)
Gg_e = Series(t=t_exp[mask_early], y=data_gamma["gh2ax"][mask_early], e=data_gamma["gh2ax_err"][mask_early])
Ng_e = Series(t=t_exp[mask_early], y=data_neutron["gh2ax"][mask_early], e=data_neutron["gh2ax_err"][mask_early])
Gp_e = Series(t=t_exp[mask_early], y=data_gamma["patm"][mask_early], e=data_gamma["patm_err"][mask_early])
Np_e = Series(t=t_exp[mask_early], y=data_neutron["patm"][mask_early], e=data_neutron["patm_err"][mask_early])

# Полное окно (для H3)
Gg_all = Series(t=t_exp, y=data_gamma["gh2ax"], e=data_gamma["gh2ax_err"])
Ng_all = Series(t=t_exp, y=data_neutron["gh2ax"], e=data_neutron["gh2ax_err"])
Gp_all = Series(t=t_exp, y=data_gamma["patm"],  e=data_gamma["patm_err"])
Np_all = Series(t=t_exp, y=data_neutron["patm"], e=data_neutron["patm_err"])

def _stack(yp, yo, eo):
    y0, e0, _ = norm_by_first(yo, eo); yp0, _, _ = norm_by_first(yp, None)
    return (yp0 - y0) / (e0 if e0 is not None else 1.0)

def res_H1(theta, model: ModelAdapter, Gg: Series, Ng: Series, Gp: Series, Np: Series):
    k_g, k_n = float(theta[0]), float(theta[1])
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_g; Pn['k_rs'] = k_n
    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, Gg.t, model.TH, model.PI_g, model.PI_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, Ng.t, model.TH, model.PI_g, model.PI_n)
    return np.r_[ _stack(gh_g, Gg.y, Gg.e), _stack(gh_n, Ng.y, Ng.e),
                  _stack(pa_g, Gp.y, Gp.e), _stack(pa_n, Np.y, Np.e) ]

def compare_H1(model, Gg, Ng, Gp, Np):
    x0 = np.array([P_BASE['k_rs'], P_BASE['k_rs']*1.1])
    lb = np.array([1e-5, 1e-5]); ub = np.array([50., 50.])
    res = least_squares(lambda th: res_H1(th, model, Gg, Ng, Gp, Np),
                        x0=x0, bounds=(lb, ub), jac="2-point", method="trf", max_nfev=6000)
    AIC = aic_of(res.fun, k_params=2)
    return res, AIC

model = ModelAdapter(P_BASE, TH, PI_G, PI_N)
resH1, AIC_H1 = compare_H1(model, Gg_e, Ng_e, Gp_e, Np_e)
print(f"[H1] AIC={AIC_H1:.2f}; k_rs^γ={resH1.x[0]:.3g}; k_rs^n={resH1.x[1]:.3g}; Δ={resH1.x[1]-resH1.x[0]:+.3g}")

def res_H2(theta, model: ModelAdapter, Gg: Series, Ng: Series, Gp: Series, Np: Series):
    k_shared, delta_w = float(theta[0]), float(theta[1])
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    Pg['k_rs'] = k_shared; Pn['k_rs'] = k_shared
    Pn['w_dc'] = Pg['w_dc'] + delta_w
    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, Gg.t, model.TH, model.PI_g, model.PI_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, Ng.t, model.TH, model.PI_g, model.PI_n)
    return np.r_[ _stack(gh_g, Gg.y, Gg.e), _stack(gh_n, Ng.y, Ng.e),
                  _stack(pa_g, Gp.y, Gp.e), _stack(pa_n, Np.y, Np.e) ]

def compare_H2(model, Gg, Ng, Gp, Np):
    x0 = np.array([P_BASE['k_rs'], 0.0])     # k_shared, Δw_dc
    lb = np.array([1e-5, -5.0]); ub = np.array([50., 5.0])
    res = least_squares(lambda th: res_H2(th, model, Gg, Ng, Gp, Np),
                        x0=x0, bounds=(lb, ub), jac="2-point", method="trf", max_nfev=6000)
    AIC = aic_of(res.fun, k_params=2)
    return res, AIC

resH2, AIC_H2 = compare_H2(model, Gg_e, Ng_e, Gp_e, Np_e)
print(f"[H2] AIC={AIC_H2:.2f}; k_rs(shared)={resH2.x[0]:.3g}; Δw_dc={resH2.x[1]:+.3g}")

best = min(AIC_H1, AIC_H2)
print(f"ΔAIC(H1–best)={AIC_H1-best:+.2f}; ΔAIC(H2–best)={AIC_H2-best:+.2f}")

def res_full(theta, model: ModelAdapter, mode: str,
             Gg: Series, Ng: Series, Gp: Series, Np: Series):
    # mode in {"H3a","H3b","H3c","base"}
    Pg, Pn = deepcopy(model.Pg), deepcopy(model.Pn)
    pi_g = model.PI_g; pi_n = model.PI_n
    if mode == "H3a":
        d_pi = float(theta[0]); pi_g = model.PI_g; pi_n = model.PI_g + d_pi
    elif mode == "H3b":
        d_k = float(theta[0]);  Pg['k_cu'] = model.Pg['k_cu']; Pn['k_cu'] = Pg['k_cu'] + d_k
    elif mode == "H3c":
        d_pi, d_k = float(theta[0]), float(theta[1])
        pi_g = model.PI_g; pi_n = model.PI_g + d_pi
        Pg['k_cu'] = model.Pg['k_cu']; Pn['k_cu'] = Pg['k_cu'] + d_k
    # base: ничего не меняем

    # предсказания на всей сетке т_exp
    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, Gg.t, model.TH, pi_g, pi_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, Ng.t, model.TH, pi_g, pi_n)

    # стек нормированных остатков (по каждому типу сигнала отдельно)
    r = np.r_[ _stack(gh_g, Gg.y, Gg.e), _stack(gh_n, Ng.y, Ng.e),
               _stack(pa_g, Gp.y, Gp.e), _stack(pa_n, Np.y, Np.e) ]
    return r

def compare_H3(model, Gg, Ng, Gp, Np):
    # base (общие π и k_cu)
    r_base = res_full([], model, "base", Gg, Ng, Gp, Np)
    AIC_base = aic_of(r_base, k_params=0)

    # H3a: только π split
    res_a = least_squares(lambda th: res_full(th, model, "H3a", Gg, Ng, Gp, Np),
                          x0=np.array([0.1]), bounds=(np.array([-0.9]), np.array([0.9])),
                          jac="2-point", method="trf", max_nfev=4000)
    AIC_a = aic_of(res_a.fun, k_params=1)

    # H3b: только k_cu split
    res_b = least_squares(lambda th: res_full(th, model, "H3b", Gg, Ng, Gp, Np),
                          x0=np.array([0.02]), bounds=(np.array([-1.0]), np.array([1.0])),
                          jac="2-point", method="trf", max_nfev=4000)
    AIC_b = aic_of(res_b.fun, k_params=1)

    # H3c: π и k_cu split вместе
    res_c = least_squares(lambda th: res_full(th, model, "H3c", Gg, Ng, Gp, Np),
                          x0=np.array([0.1, 0.02]),
                          bounds=(np.array([-0.9, -1.0]), np.array([0.9, 1.0])),
                          jac="2-point", method="trf", max_nfev=5000)
    AIC_c = aic_of(res_c.fun, k_params=2)

    best = min(AIC_base, AIC_a, AIC_b, AIC_c)
    print("\n=== H3 (full 0.5–24 h) ===")
    print(f"BASE (shared π,k_cu): AIC={AIC_base:.2f}")
    print(f"H3a (π split)       : AIC={AIC_a:.2f},  Δπ={res_a.x[0]:+.3f}")
    print(f"H3b (k_cu split)    : AIC={AIC_b:.2f},  Δk_cu={res_b.x[0]:+.3f}")
    print(f"H3c (π & k_cu split): AIC={AIC_c:.2f},  Δπ={res_c.x[0]:+.3f}, Δk_cu={res_c.x[1]:+.3f}")
    print(f"ΔAIC(base–best)={AIC_base-best:+.2f}; ΔAIC(H3a–best)={AIC_a-best:+.2f}; ΔAIC(H3b–best)={AIC_b-best:+.2f}; ΔAIC(H3c–best)={AIC_c-best:+.2f}")

compare_H3(model, Gg_all, Ng_all, Gp_all, Np_all)

import numpy as np
rng = np.random.default_rng(12345)

def _gen_boot_obs_early(model, Gg, Ng, Gp, Np, mode="H1", theta=None):
    """
    Генерируем синтетические наблюдения для 0.5–2 ч вокруг текущих fit-предсказаний
    с добавлением гауссовского шума по экспериментальным σ. Нормировка НЕ нужна — шумим в «сырых» единицах.
    mode: "H1" -> theta=[k_g, k_n]; "H2" -> theta=[k_shared, delta_w]
    """
    Pg = deepcopy(model.Pg); Pn = deepcopy(model.Pn)
    if mode == "H1":
        Pg['k_rs'] = float(theta[0]); Pn['k_rs'] = float(theta[1])
    elif mode == "H2":
        k_shared, dwdc = float(theta[0]), float(theta[1])
        Pg['k_rs'] = k_shared; Pn['k_rs'] = k_shared
        Pn['w_dc'] = Pg['w_dc'] + dwdc
    else:
        raise ValueError

    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, Gg.t, model.TH, model.PI_g, model.PI_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, Ng.t, model.TH, model.PI_g, model.PI_n)

    # добавляем шум ~ N(0, sigma), где sigma берём из твоих *_err
    yGg = gh_g + rng.normal(0.0, _ensure_1d(Gg.e))
    yNg = gh_n + rng.normal(0.0, _ensure_1d(Ng.e))
    yGp = pa_g + rng.normal(0.0, _ensure_1d(Gp.e))
    yNp = pa_n + rng.normal(0.0, _ensure_1d(Np.e))
    return Series(Gg.t, yGg, Gg.e), Series(Ng.t, yNg, Ng.e), Series(Gp.t, yGp, Gp.e), Series(Np.t, yNp, Np.e)

def bootstrap_H1(model, Gg, Ng, Gp, Np, B=500):
    # сначала получаем МLE на реальных данных
    x0 = np.array([P_BASE['k_rs'], P_BASE['k_rs']*1.1])
    res0 = least_squares(lambda th: res_H1(th, model, Gg, Ng, Gp, Np),
                         x0=x0, bounds=(np.array([1e-5,1e-5]), np.array([50.,50.])),
                         jac="2-point", method="trf", max_nfev=6000)
    AIC0 = aic_of(res0.fun, k_params=2)
    k_g0, k_n0 = res0.x
    d_kn0 = k_n0 - k_g0

    boots = []
    for b in range(B):
        # генерим bootstrap-датасет вокруг текущего fit
        Gg_b, Ng_b, Gp_b, Np_b = _gen_boot_obs_early(model, Gg, Ng, Gp, Np, mode="H1", theta=res0.x)
        # рефит на бутстрэп-данных
        res_b = least_squares(lambda th: res_H1(th, model, Gg_b, Ng_b, Gp_b, Np_b),
                              x0=res0.x, bounds=(np.array([1e-5,1e-5]), np.array([50.,50.])),
                              jac="2-point", method="trf", max_nfev=4000)
        AIC_b = aic_of(res_b.fun, k_params=2)
        k_g, k_n = res_b.x
        boots.append((k_g, k_n, k_n - k_g, AIC_b))
    boots = np.array(boots)
    return {"theta0": (k_g0, k_n0, d_kn0, AIC0), "boots": boots}

def bootstrap_H2(model, Gg, Ng, Gp, Np, B=500):
    x0 = np.array([P_BASE['k_rs'], 0.0])
    res0 = least_squares(lambda th: res_H2(th, model, Gg, Ng, Gp, Np),
                         x0=x0, bounds=(np.array([1e-5,-5.0]), np.array([50.,5.0])),
                         jac="2-point", method="trf", max_nfev=6000)
    AIC0 = aic_of(res0.fun, k_params=2)
    kshared0, dwdc0 = res0.x

    boots = []
    for b in range(B):
        Gg_b, Ng_b, Gp_b, Np_b = _gen_boot_obs_early(model, Gg, Ng, Gp, Np, mode="H2", theta=res0.x)
        res_b = least_squares(lambda th: res_H2(th, model, Gg_b, Ng_b, Gp_b, Np_b),
                              x0=res0.x, bounds=(np.array([1e-5,-5.0]), np.array([50.,5.0])),
                              jac="2-point", method="trf", max_nfev=4000)
        AIC_b = aic_of(res_b.fun, k_params=2)
        boots.append((res_b.x[0], res_b.x[1], AIC_b))
    boots = np.array(boots)
    return {"theta0": (kshared0, dwdc0, AIC0), "boots": boots}

def summarize_percentile(arr, q=(2.5, 97.5)):
    lo, hi = np.percentile(arr, q[0]), np.percentile(arr, q[1])
    med = np.percentile(arr, 50)
    return med, lo, hi

model = ModelAdapter(P_BASE, TH, PI_G, PI_N)

B = 500  # можно 1000, если время позволяет
H1b = bootstrap_H1(model, Gg_e, Ng_e, Gp_e, Np_e, B=B)
H2b = bootstrap_H2(model, Gg_e, Ng_e, Gp_e, Np_e, B=B)

print("\n[BOOT H1] Δk_rs = k_n - k_g")
med, lo, hi = summarize_percentile(H1b["boots"][:,2])
print(f"median={med:.3g}, 95% CI [{lo:.3g}; {hi:.3g}], MLE={H1b['theta0'][2]:.3g}")
p_sign = np.mean(H1b["boots"][:,2] > 0.0)
print(f"P(Δk_rs>0)≈{p_sign:.3f}  -> {'поддержка H1' if p_sign>0.95 else 'нет'}")

print("\n[BOOT H2] Δw_dc (neutron - gamma)")
med, lo, hi = summarize_percentile(H2b["boots"][:,1])
print(f"median={med:.3g}, 95% CI [{lo:.3g}; {hi:.3g}], MLE={H2b['theta0'][1]:.3g}")

def _gen_boot_obs_full(model, Gg, Ng, Gp, Np, mode="H3c", theta=None):
    Pg = deepcopy(model.Pg); Pn = deepcopy(model.Pn)
    pi_g = model.PI_g; pi_n = model.PI_n
    if mode == "H3a":
        d_pi = float(theta[0]); pi_n = pi_g + d_pi
    elif mode == "H3b":
        d_k = float(theta[0]); Pn['k_cu'] = Pg['k_cu'] + d_k
    elif mode == "H3c":
        d_pi, d_k = float(theta[0]), float(theta[1])
        pi_n = pi_g + d_pi; Pn['k_cu'] = Pg['k_cu'] + d_k
    elif mode == "base":
        pass
    else:
        raise ValueError

    gh_g, pa_g = simulate_Cplus(Pg, Pn, LET_GAMMA,  DOSE_Gy, Gg.t, model.TH, pi_g, pi_n)
    gh_n, pa_n = simulate_Cplus(Pg, Pn, LET_NEUTRON,DOSE_Gy, Ng.t, model.TH, pi_g, pi_n)

    yGg = gh_g + rng.normal(0.0, _ensure_1d(Gg.e))
    yNg = gh_n + rng.normal(0.0, _ensure_1d(Ng.e))
    yGp = pa_g + rng.normal(0.0, _ensure_1d(Gp.e))
    yNp = pa_n + rng.normal(0.0, _ensure_1d(Np.e))
    return Series(Gg.t, yGg, Gg.e), Series(Ng.t, yNg, Ng.e), Series(Gp.t, yGp, Gp.e), Series(Np.t, yNp, Np.e)

def fit_H3_mode(model, Gg, Ng, Gp, Np, mode):
    if mode == "base":
        r = res_full([], model, "base", Gg, Ng, Gp, Np); return None, aic_of(r, k_params=0)
    if mode == "H3a":
        res = least_squares(lambda th: res_full(th, model, "H3a", Gg, Ng, Gp, Np),
                            x0=np.array([0.1]), bounds=(np.array([-0.9]), np.array([0.9])),
                            jac="2-point", method="trf", max_nfev=4000)
        return res, aic_of(res.fun, k_params=1)
    if mode == "H3b":
        res = least_squares(lambda th: res_full(th, model, "H3b", Gg, Ng, Gp, Np),
                            x0=np.array([0.02]), bounds=(np.array([-1.0]), np.array([1.0])),
                            jac="2-point", method="trf", max_nfev=4000)
        return res, aic_of(res.fun, k_params=1)
    if mode == "H3c":
        res = least_squares(lambda th: res_full(th, model, "H3c", Gg, Ng, Gp, Np),
                            x0=np.array([0.1,0.02]),
                            bounds=(np.array([-0.9,-1.0]), np.array([0.9,1.0])),
                            jac="2-point", method="trf", max_nfev=5000)
        return res, aic_of(res.fun, k_params=2)

def bootstrap_H3c(model, Gg, Ng, Gp, Np, B=500):
    # MLE на данных
    res0, AIC0 = fit_H3_mode(model, Gg, Ng, Gp, Np, "H3c")
    d_pi0, d_k0 = res0.x
    boots = []
    for b in range(B):
        Gg_b, Ng_b, Gp_b, Np_b = _gen_boot_obs_full(model, Gg, Ng, Gp, Np, mode="H3c", theta=res0.x)
        res_b, AIC_b = fit_H3_mode(model, Gg_b, Ng_b, Gp_b, Np_b, "H3c")
        boots.append((res_b.x[0], res_b.x[1], AIC_b))
    boots = np.array(boots)
    return {"theta0": (d_pi0, d_k0, AIC0), "boots": boots}
H3c_b = bootstrap_H3c(model, Gg_all, Ng_all, Gp_all, Np_all, B=500)
print("\n[BOOT H3c] Δπ, Δk_cu")
med_dpi, lo_dpi, hi_dpi = summarize_percentile(H3c_b["boots"][:,0])
med_dk,  lo_dk,  hi_dk  = summarize_percentile(H3c_b["boots"][:,1])
print(f"Δπ: median={med_dpi:.3g}, 95% CI [{lo_dpi:.3g}; {hi_dpi:.3g}], MLE={H3c_b['theta0'][0]:.3g}")
print(f"Δk_cu: median={med_dk:.3g}, 95% CI [{lo_dk:.3g}; {hi_dk:.3g}], MLE={H3c_b['theta0'][1]:.3g}")
