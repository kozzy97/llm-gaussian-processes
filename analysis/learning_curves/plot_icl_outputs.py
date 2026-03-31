from collections import defaultdict
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import t as t_dist
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import GPy

gp_json = "/llm-gaussian-processes/gp_dataset/dataset_1d_mae_se_matern_ls_1_8.json"

llm_csvs = [
    "/llm-gaussian-processes/model_prediction_data/BASE/Qwen3-8B-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/Qwen3-14B-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/Qwen3-32B-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/gemma-3-4b-it-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/gemma-3-12b-it-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/gemma-3-27b-it-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/Llama-3.1-8B-Instruct-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/Llama-3.2-3B-Instruct-unsloth-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/Meta-Llama-3.1-70B-Instruct-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/mistral-7b-instruct-v0.3-bnb-4bit.csv",
    "/llm-gaussian-processes/model_prediction_data/BASE/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit.csv",
]

plot_llm = True
llm_smoothing = "none"
out_dir = Path("/llm-gaussian-processes/analysis/plots/learning_curves_1d")
out_dir.mkdir(parents=True, exist_ok=True)

max_iter = 50
x_start  = 2
noise    = 0.001
noise_levels  = [noise]
domain_L      = 29.0
kernels       = ["SE_ls1", "SE_ls8", "Matern_05_ls1", "Matern_05_ls8"]

plt.rcParams.update({
    'font.size': 26, 'axes.titlesize': 24, 'axes.labelsize': 26,
    'legend.fontsize': 22, 'xtick.labelsize': 26, 'ytick.labelsize': 26
})

def smooth_causal_ema(y, span=10):
    y = np.asarray(y, float)
    if len(y) == 0 or span <= 1: return y.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def isotonic_nonincreasing(y, w=None):
    y = np.asarray(y, float)
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w, float)
    x = -y
    v = x.copy()
    wn = w.copy()
    n = len(v)
    i = 0
    while i < n - 1:
        if v[i] > v[i+1] + 1e-15:
            j = i
            while j >= 0 and v[j] > v[j+1] + 1e-15:
                total_w = wn[j] + wn[j+1]
                avg = (wn[j]*v[j] + wn[j+1]*v[j+1]) / total_w
                v[j] = avg
                wn[j] = total_w
                # delete j+1 by shifting left
                v = np.delete(v, j+1)
                wn = np.delete(wn, j+1)
                n -= 1
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    x = x.copy()
    blocks = [[0]]
    for i in range(1, len(x)):
        blocks.append([i])
        while len(blocks) >= 2:
            last = blocks[-1]
            prev = blocks[-2]
            m_last = np.average(x[last], weights=w[last])
            m_prev = np.average(x[prev], weights=w[prev])
            if m_prev > m_last + 1e-15:
                blocks[-2] = prev + last
                blocks.pop()
            else:
                break
    v_fit = np.empty_like(x)
    for blk in blocks:
        m = np.average(x[blk], weights=w[blk])
        v_fit[blk] = m
    return -v_fit

def apply_llm_smoothing(means, ns, method="isotonic"):
    if method == "isotonic":
        return isotonic_nonincreasing(means, w=np.where(np.isnan(ns), 0.0, ns))
    elif method == "ema":
        return smooth_causal_ema(means, span=10)
    else:
        return means

def mse_to_mae(mse): return np.sqrt(2.0/np.pi) * np.sqrt(np.maximum(mse, 0.0))

def _nn_pdf_distance(n, L):
    if n < 2: raise ValueError("Need at least two points.")
    def fD(d):
        base = 1.0 - 2.0 * d / L
        return 0.0 if base <= 0.0 else (2.0 * (n - 1) / L) * (base ** (n - 2))
    return fD

def _var_given_d_se(d, ell, sigma_eps2=0.001, sigma_f2=1.0):
    k = sigma_f2 * np.exp(-(d * d) / (2.0 * ell * ell))
    v = 2.0 * (sigma_f2 + sigma_eps2) - 2.0 * k
    return max(v, 0.0)

def _var_given_d_m12(d, ell, sigma_eps2=0.001, sigma_f2=1.0):
    k = sigma_f2 * np.exp(-d / ell)
    v = 2.0 * (sigma_f2 + sigma_eps2) - 2.0 * k
    return max(v, 0.0)

def knn_abs_moments(n, ell, kernel_type="SE", L=domain_L, sigma_eps2=noise, sigma_f2=1.0):
    fD = _nn_pdf_distance(n, L)
    var_d = _var_given_d_se if kernel_type == "SE" else _var_given_d_m12
    E_sqrtv, _ = quad(lambda d: np.sqrt(var_d(d, ell, sigma_eps2, sigma_f2)) * fD(d), 0.0, L/2.0, limit=200)
    E_v,      _ = quad(lambda d:        var_d(d, ell, sigma_eps2, sigma_f2)  * fD(d), 0.0, L/2.0, limit=200)
    mean_abs = np.sqrt(2.0 / np.pi) * E_sqrtv
    var_abs  = max(E_v - mean_abs**2, 0.0)
    return mean_abs, var_abs

def eigvals_SE_1d(length_scale, L=1.0, qmax=2000):
    qs = np.arange(0, qmax+1); w = 2*np.pi*qs/L
    S = np.sqrt(2*np.pi) * length_scale * np.exp(-0.5*(length_scale*w)**2)
    Λ = S.copy(); Λ[1:] *= 2.0
    Λ = Λ / L
    Λ = Λ / np.sum(Λ)
    return Λ

def eigvals_Matern12_1d(length_scale, L=1.0, qmax=2000):
    qs = np.arange(0, qmax+1); w = 2*np.pi*qs/L
    S = 2*length_scale / (1.0 + (length_scale*w)**2)
    Λ = S.copy(); Λ[1:] *= 2.0
    Λ = Λ / L
    Λ = Λ / np.sum(Λ)
    return Λ

def make_eigs(kname, ell, L=domain_L, qmax=2000):
    return eigvals_SE_1d(ell, L, qmax) if kname.startswith("SE") else eigvals_Matern12_1d(ell, L, qmax)

def gp_uc_mse(n, eigvals, sigma2):
    A = np.asarray(eigvals, float)
    if n <= 0: return float(np.sum(A))
    def F(nprime):  # n' + Σ log(1 + (n'/σ^2) λ) - n = 0
        return nprime + np.sum(np.log1p((nprime/sigma2)*A)) - n
    hi = max(n, 1e-9); lo = 0.0
    while F(hi) < 0:
        hi *= 2
        if hi > 1e9: break
    nprime = brentq(F, lo+1e-12, hi)
    eps = np.sum(1.0 / (1.0/A + (nprime/sigma2)))
    return float(eps)

def gp_lc_mse(n, eigvals, sigma2, tol=1e-10, maxit=100):
    A = np.asarray(eigvals, float)
    if n <= 0: return float(np.sum(A))
    nprime = n * sigma2 / (sigma2 + 1.0)  # init
    for _ in range(maxit):
        eps = np.sum(1.0 / (1.0/A + (nprime/sigma2)))
        nprime_new = n * sigma2 / (sigma2 + eps)
        if abs(nprime_new - nprime) <= tol * max(1.0, nprime): break
        nprime = nprime_new
    eps = np.sum(1.0 / (1.0/A + (nprime/sigma2)))
    return float(eps)

def collect_data(prompt: str) -> tuple[list, list]:
    lines = [ln.strip() for ln in prompt.split("\n") if ln.strip()]
    xs, ys, ids = [], [], []
    for i, line in enumerate(lines):
        m = re.match(r"X: ([\d\.]+), Y: (-?[\d\.]+)", line)
        if not m:
            continue
        x_val, y_val = m.groups()
        xs.append(float(x_val))
        ys.append(float(y_val))
        ids.append(i)
    assert len(xs) == len(ys), f"len(xs)={len(xs)} != len(ys)={len(ys)}"
    return xs, ys, ids

def _get_kernel(s: str):
    if "SE" in s:
        return GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0) if "_ls1" in s else GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=8.0)
    elif "Matern_05" in s:
        return GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=1.0) if "_ls1" in s else GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=8.0)
    else:
        raise ValueError(f"Unknown kernel string: {s}")

with open(gp_json, "r") as f:
    gp_raw = json.load(f)

abs_errs_all = []
for e in gp_raw:
    xs, ys, ids = collect_data(e.get("text"))
    xs, ys, ids = xs[:max_iter], ys[:max_iter], ids[:max_iter]
    xs, ys = [float(np.asarray(x).squeeze()) for x in xs], [float(np.asarray(y).squeeze()) for y in ys]
    k_name = str(e.get("kernel"))
    print(k_name)
    k = _get_kernel(k_name)
    nz = float(e.get("noise"))
    abs_errs = []
    for idx, (x_test, y_true) in enumerate(zip(xs, ys)):
        x_test_arr = np.array([[x_test]])
        if idx == 0:
            mean_pred = 0.0
        else:
            X_train = np.array(xs[:idx])[:, None]   # shape (i,1)
            Y_train = np.array(ys[:idx])[:, None]
            k = k.copy()
            gp = GPy.models.GPRegression(X_train, Y_train, kernel=k, normalizer=True) # normalizer for zero-mean prior
            gp.Gaussian_noise.variance = nz
            _ = gp.Gaussian_noise.variance.fix()
            mean_pred, var_pred = gp.predict(x_test_arr)
            mean_pred = mean_pred.item()
        abs_errs.append(abs(mean_pred - y_true))
    abs_errs_all.append((k_name, nz, abs_errs))

acc = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "n": 0}))

for k_name, nz, abs_errs in abs_errs_all:
    for idx, err in enumerate(abs_errs):
        acc[(k_name, nz)][idx]["sum"] += float(err)
        acc[(k_name, nz)][idx]["n"]   += 1

mae_by_group = {}
for key, idx_map in acc.items():
    max_idx = max(idx_map)  # largest idx seen for this group
    maes = []
    for idx in range(max_idx + 1):
        if idx in idx_map and idx_map[idx]["n"] > 0:
            maes.append(idx_map[idx]["sum"] / idx_map[idx]["n"])
        else:
            maes.append(np.nan)
    mae_by_group[key] = maes

gp_rows = defaultdict(list)
for k_name, nz, abs_errs in abs_errs_all:
    gp_rows[(k_name, nz)].append(np.array(abs_errs, float)[:max_iter])

rng = np.random.default_rng(0)
B = 1000
gp_boot_ci = {}
for key, rows in gp_rows.items():
    R = len(rows)
    A = np.full((R, max_iter), np.nan)
    for i, r in enumerate(rows):
        A[i, :len(r)] = r
    boot = np.empty((B, max_iter))
    for b in range(B):
        idx = rng.integers(0, R, size=R)
        boot[b] = np.nanmean(A[idx], axis=0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    gp_boot_ci[key] = (lo, hi)


available_kernels = kernels.copy()

def assign_function_ids_by_order(df, max_iter=50):
    df = df.copy(); df["__row__"] = np.arange(len(df))
    out = []
    for _, sub in df.groupby(["kernel", "noise"], sort=False):
        sub = sub.sort_values("__row__")
        func_id = (np.arange(len(sub)) // max_iter) + 1
        tmp = sub.copy(); tmp["FunctionID"] = func_id
        sizes = tmp.groupby("FunctionID")["Trial"].nunique()
        ok = sizes.min() == max_iter and sizes.max() == max_iter
        if not ok:
            starts = (sub["Trial"] == 1).astype(int).values
            if starts.size == 0: continue
            if starts[0] == 0: starts[0] = 1
            func_id = np.cumsum(starts)
        sub = sub.copy(); sub["FunctionID"] = func_id
        out.append(sub)
    out = pd.concat(out, ignore_index=True)
    out.drop(columns="__row__", inplace=True)
    return out

def llm_prefix_stats(df_model_kernel, max_iter=50, smoothing="isotonic"):
    pivot = df_model_kernel.pivot(index="FunctionID", columns="Trial", values="Absolute_Error")
    complete = pivot.columns.intersection(range(1, max_iter+1))
    pivot = pivot[complete].dropna(axis=0)
    means, stds, ns = [], [], []
    for t in range(1, max_iter+1):
        col = pivot.loc[:, 1:t]
        ok_rows = col.notna().all(axis=1)
        vals = pivot.loc[ok_rows, t].astype(float)
        n = int(vals.count())
        ns.append(n)
        means.append(float(vals.mean()) if n > 0 else np.nan)
        stds.append(float(vals.std(ddof=1)) if n > 1 else 0.0)
    means = np.array(means); stds = np.array(stds); ns = np.array(ns)
    tcrit = t_dist.ppf(0.975, df=np.maximum(ns-1,1))
    se = np.where(ns>0, stds/np.sqrt(ns), np.nan)
    
    mean_disp_full = apply_llm_smoothing(means, ns, method=smoothing)
    
    lo_full = mean_disp_full - tcrit * se
    hi_full = mean_disp_full + tcrit * se
    
    sl = slice(x_start-1, max_iter)
    return mean_disp_full[sl], lo_full[sl], hi_full[sl], ns[sl]

for llm_csv in llm_csvs:
    print(f"\n{'='*60}")
    print(f"Processing: {llm_csv}")
    print('='*60)
    
    model_name = llm_csv.split("/")[-1].replace(".csv", "")
    
    if plot_llm:
        df_llm = pd.read_csv(llm_csv)
        df_llm = df_llm[df_llm["noise"].astype(float).isin(noise_levels)].copy()
        df_llm["Trial"] = df_llm["Trial"].astype(int)
        df_llm = assign_function_ids_by_order(df_llm, max_iter=max_iter)

    x = np.arange(x_start, max_iter + 1)

    for k_name in available_kernels:
        print(f"Plotting {k_name}, noise={noise} ...")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        y_emp_full = mae_by_group[(k_name, noise)]
        y_emp = np.asarray(y_emp_full)[x_start-1:max_iter]
        ax.plot(x, y_emp, color="#9AA0A6", lw=3, label="GP")
        
        lo_emp_full, hi_emp_full = gp_boot_ci[(k_name, noise)]
        lo_emp = lo_emp_full[x_start-1:max_iter]
        hi_emp = hi_emp_full[x_start-1:max_iter]
        ax.fill_between(x, lo_emp, hi_emp, color="#9AA0A6", alpha=0.20, linewidth=0)
        
        if plot_llm:
            sub = df_llm[(df_llm["kernel"] == k_name) & (df_llm["noise"].astype(float) == noise)]
            if len(sub) > 0:
                y_llm, lo_llm, hi_llm, ns = llm_prefix_stats(sub, max_iter=max_iter, smoothing=llm_smoothing)
                ax.plot(x, y_llm, color="#4285F4", lw=3, label=f"LLM ({llm_smoothing})")
                ax.fill_between(x, lo_llm, hi_llm, color="#4285F4", alpha=0.20, linewidth=0)
                print(f"  LLM: functions used ~{int(np.nanmax(ns))} (prefix-complete)")
            else:
                print("  [warn] No LLM rows for this kernel/noise.")
        
        try:
            ell = float(k_name.split("_ls")[-1])
        except Exception:
            ell = 1.0
        kernel_type = "SE" if k_name.startswith("SE") else "Matern_05"
        R_eff = 200
        means_nn, se_nn = [], []
        for n in range(1, max_iter + 1):
            if n == 1:
                means_nn.append(0.0); se_nn.append(0.0); continue
            m_abs, v_abs = knn_abs_moments(n, ell, kernel_type=kernel_type, L=domain_L, sigma_eps2=noise, sigma_f2=1.0)
            means_nn.append(m_abs)
            se_nn.append(np.sqrt(v_abs) / np.sqrt(max(R_eff, 1)))
        y_nn = np.array(means_nn)[x_start-1:max_iter]
        se_nn = np.array(se_nn)[x_start-1:max_iter]
        tcrit_nn = t_dist.ppf(0.975, df=max(R_eff - 1, 1))
        ax.plot(x, y_nn, color='black', lw=3, label="1-NN")
        ax.fill_between(x, y_nn - tcrit_nn*se_nn, y_nn + tcrit_nn*se_nn, color='black', alpha=0.0, linewidth=0)
        
        eigs = make_eigs(k_name, ell, L=domain_L, qmax=2000)
        eps_uc_mse = np.array([gp_uc_mse(n, eigs, sigma2=noise) for n in range(1, max_iter+1)])
        eps_lc_mse = np.array([gp_lc_mse(n, eigs, sigma2=noise) for n in range(1, max_iter+1)])
        eps_uc_mae = mse_to_mae(eps_uc_mse)[x_start-1:max_iter]
        eps_lc_mae = mse_to_mae(eps_lc_mse)[x_start-1:max_iter]
        
        ax.set_xlim(x_start - 0.5, max_iter + 0.5)
        ax.set_ylim(-0.02, 1.00)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel("Trial")
        ax.set_ylabel("MAE")
        base_k = "SE" if k_name.startswith("SE") else "Matérn 1/2"
        lam    = k_name.split("_ls")[-1]
        ax.set_title(rf"{base_k}, $\lambda$ = {lam}, noise = {noise}")
        
        handles = [
            Line2D([0], [0], color="#9AA0A6", lw=3, label="GP"),
            Line2D([0], [0], color='black', lw=3, label="1-NN"),
        ]
        if plot_llm:
            handles.insert(1, Line2D([0], [0], color="#4285F4", lw=3, label=f"LLM"))
        ax.legend(handles=handles, handlelength=1.0, handletextpad=0.5, loc="upper right")
        
        fig.tight_layout()
        out_file = out_dir / f"{model_name}_LC_UC_mae_{llm_smoothing}_{k_name}_noise={noise}_L={int(domain_L)}.png"
        plt.savefig(out_file, dpi=300)
        plt.show()
        print(f"Saved: {out_file}")

        save_dir = Path(
            "/llm-gaussian-processes/analysis/curves_saved_data"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i, t in enumerate(x):
            row = {
                "model_name": model_name,
                "kernel": k_name,
                "noise": noise,
                "trial": int(t),

                "llm_mean": y_llm[i] if plot_llm else np.nan,
                "llm_lo":   lo_llm[i] if plot_llm else np.nan,
                "llm_hi":   hi_llm[i] if plot_llm else np.nan,

                "gp_mean": y_emp[i],
                "gp_lo":   lo_emp[i],
                "gp_hi":   hi_emp[i],

                "nn_mean": y_nn[i],
                "nn_se":   se_nn[i],
            }
            records.append(row)

        df_out = pd.DataFrame(records)

        csv_path = save_dir / (
            f"{model_name}_1D_{k_name}_noise={noise}_L={int(domain_L)}.csv"
        )
        df_out.to_csv(csv_path, index=False)
        print(f"  → Saved curve data to CSV: {csv_path}")
