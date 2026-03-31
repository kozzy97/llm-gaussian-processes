from collections import defaultdict
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.stats import t as t_dist
from scipy.special import gamma
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import GPy

gp_json_paths = {
    2: "/llm-gaussian-processes/gp_dataset/dataset_2d_mae_se_matern_ls_1_8.json",
    3: "/llm-gaussian-processes/gp_dataset/dataset_3d_mae_se_matern_ls_1_8.json",
    4: "/llm-gaussian-processes/gp_dataset/dataset_4d_mae_se_matern_ls_1_8.json"
}

llm_csvs = [
    "/llm-gaussian-processes/model_prediction_data/MULTI-DIM/BASE/Qwen3-14B-unsloth-bnb-4bit_dataset_2d_mae_se_matern_ls_1_8_results.csv",
    "/llm-gaussian-processes/model_prediction_data/MULTI-DIM/BASE/Qwen3-14B-unsloth-bnb-4bit_dataset_3d_mae_se_matern_ls_1_8_results.csv",
    "/llm-gaussian-processes/model_prediction_data/MULTI-DIM/BASE/Qwen3-14B-unsloth-bnb-4bit_dataset_4d_mae_se_matern_ls_1_8_results.csv",
    # "/llm-gaussian-processes/model_prediction_data/MULTI-DIM/BASE/Qwen3-32B-unsloth-bnb-4bit_dataset_2d_mae_se_matern_ls_1_8_results.csv",
    # "/llm-gaussian-processes/model_prediction_data/MULTI-DIM/BASE/Qwen3-32B-unsloth-bnb-4bit_dataset_3d_mae_se_matern_ls_1_8_results.csv",
    # "/llm-gaussian-processes/model_prediction_data/MULTI-DIM/BASE/Qwen3-32B-unsloth-bnb-4bit_dataset_4d_mae_se_matern_ls_1_8_results.csv",
]

plot_llm = True
llm_smoothing = "none"
out_dir = Path("/llm-gaussian-processes/analysis/plots/learning_curves_multidim")
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
    """Causal EMA; no look-ahead."""
    y = np.asarray(y, float)
    if len(y) == 0 or span <= 1: 
        return y.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def isotonic_nonincreasing(y, w=None):
    """Pool-Adjacent-Violators Algorithm (PAVA) for non-increasing fit."""
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
    """Apply display-time smoothing to LLM mean curve only."""
    if method == "isotonic":
        return isotonic_nonincreasing(means, w=np.where(np.isnan(ns), 0.0, ns))
    elif method == "ema":
        return smooth_causal_ema(means, span=10)
    else:
        return means

def _nn_pdf_distance_multidim(n, L, d_dim):
    """PDF of nearest neighbor distance in d-dimensional hypercube [0,L]^d."""
    if n < 2: 
        raise ValueError("Need at least two points.")
    V_total = L ** d_dim
    def fD(r):
        V_ball = (np.pi ** (d_dim / 2.0)) / gamma(d_dim / 2.0 + 1) * (r ** d_dim)
        p_outside = max(0.0, 1.0 - V_ball / V_total)
        if p_outside <= 0 or p_outside >= 1: 
            return 0.0
        surface_area = d_dim * V_ball / r if r > 0 else 0.0
        pdf = (n - 1) * surface_area / V_total * (p_outside ** (n - 2))
        return max(0.0, pdf)
    return fD

def _var_given_d_se_multidim(d, ell, sigma_eps2=0.001, sigma_f2=1.0):
    k = sigma_f2 * np.exp(-(d * d) / (2.0 * ell * ell))
    v = 2.0 * (sigma_f2 + sigma_eps2) - 2.0 * k
    return max(v, 0.0)

def _var_given_d_m12_multidim(d, ell, sigma_eps2=0.001, sigma_f2=1.0):
    k = sigma_f2 * np.exp(-d / ell)
    v = 2.0 * (sigma_f2 + sigma_eps2) - 2.0 * k
    return max(v, 0.0)

def knn_abs_moments_multidim(n, ell, d_dim, kernel_type="SE", L=29.0, sigma_eps2=0.001, sigma_f2=1.0):
    fD = _nn_pdf_distance_multidim(n, L, d_dim)
    var_d = _var_given_d_se_multidim if kernel_type == "SE" else _var_given_d_m12_multidim
    max_dist = L * np.sqrt(d_dim)
    E_sqrtv, _ = quad(lambda d: np.sqrt(var_d(d, ell, sigma_eps2, sigma_f2)) * fD(d), 0.0, max_dist, limit=200)
    E_v, _ = quad(lambda d: var_d(d, ell, sigma_eps2, sigma_f2) * fD(d), 0.0, max_dist, limit=200)
    mean_abs = np.sqrt(2.0 / np.pi) * E_sqrtv
    var_abs = max(E_v - mean_abs**2, 0.0)
    return mean_abs, var_abs

def collect_gp_data_multidim(prompt: str, d_dim: int) -> tuple[list, list]:
    """Parses X vectors and Y values from the JSON text prompt."""
    lines = [ln.strip() for ln in prompt.split("\n") if ln.strip()]
    xs_all, ys_all = [], []
    
    for line in lines:
        # Regex to find all X coords: X0: 1.2, X1: 3.4
        # and Y coord: Y: 5.6
        x_matches = re.findall(r"X\d+:\s*([-+]?\d*\.?\d+)", line)
        y_match = re.search(r"Y:\s*([-+]?\d*\.?\d+)", line)
        
        if len(x_matches) == d_dim and y_match:
            row_coords = [float(v) for v in x_matches]
            y_val = float(y_match.group(1))
            xs_all.append(row_coords)
            ys_all.append(y_val)
            
    return xs_all, ys_all

def _get_gpy_kernel(s: str, input_dim: int):
    if "SE" in s:
        ls = 1.0 if "_ls1" in s else 8.0
        return GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=ls)
    elif "Matern_05" in s:
        ls = 1.0 if "_ls1" in s else 8.0
        return GPy.kern.Exponential(input_dim=input_dim, variance=1.0, lengthscale=ls)
    else:
        raise ValueError(f"Unknown kernel string: {s}")

def compute_empirical_gp_stats(json_path, d_dim, max_iter):
    if not Path(json_path).exists():
        print(f"Warning: GP JSON not found: {json_path}")
        return {}

    print(f"Loading GP JSON for {d_dim}D: {json_path} ...")
    with open(json_path, "r") as f:
        gp_raw = json.load(f)
    
    abs_errs_all = []

    for i, e in enumerate(gp_raw):
        if i % 10 == 0: 
            print(f"  Processing GP function {i+1}/{len(gp_raw)}...", end='\r')
        
        k_name = str(e.get("kernel"))
        nz = float(e.get("noise"))
        
        xs, ys = collect_gp_data_multidim(e.get("text"), d_dim)
        
        xs = np.array(xs[:max_iter])
        ys = np.array(ys[:max_iter])
        
        if len(ys) == 0: 
            continue

        k = _get_gpy_kernel(k_name, input_dim=d_dim)
        
        curr_abs_errs = []
        
        for idx in range(len(ys)):
            x_test = xs[idx:idx+1]
            y_true = ys[idx]
            
            if idx == 0:
                mean_pred = 0.0
            else:
                X_train = xs[:idx]
                Y_train = ys[:idx, None]
                k_inst = k.copy()
                
                gp = GPy.models.GPRegression(X_train, Y_train, kernel=k_inst, normalizer=True) # normalizer for zero-mean prior
                gp.Gaussian_noise.variance = nz
                gp.Gaussian_noise.variance.fix()
                
                mean_pred_arr, _ = gp.predict(x_test)
                mean_pred = mean_pred_arr.item()
            
            curr_abs_errs.append(abs(mean_pred - y_true))
            
        abs_errs_all.append((k_name, nz, curr_abs_errs))
    
    print("\n  Done calculating GP regressions. Bootstrapping CIs...")

    gp_rows = defaultdict(list)
    for k_name, nz, abs_errs in abs_errs_all:
        row = np.full(max_iter, np.nan)
        row[:len(abs_errs)] = abs_errs
        gp_rows[(k_name, nz)].append(row)

    results = {}
    rng = np.random.default_rng(42)
    B = 1000
    
    for key, rows in gp_rows.items():
        rows_arr = np.array(rows) # (R, max_iter)
        R = rows_arr.shape[0]
        if R == 0: 
            continue
        
        boot_means = np.empty((B, max_iter))
        for b in range(B):
            idx = rng.integers(0, R, size=R)
            boot_means[b] = np.nanmean(rows_arr[idx], axis=0)
            
        mean_curve = np.nanmean(rows_arr, axis=0)
        lo = np.nanpercentile(boot_means, 2.5, axis=0)
        hi = np.nanpercentile(boot_means, 97.5, axis=0)
        
        results[key] = (mean_curve, lo, hi)
        
    return results

def assign_function_ids_by_order(df, max_iter=50):
    """Assign function IDs based on groups of max_iter trials."""
    df = df.copy()
    df["__row__"] = np.arange(len(df))
    out = []
    for _, sub in df.groupby(["kernel", "noise", "dimension"], sort=False):
        sub = sub.sort_values("__row__")
        func_id = (np.arange(len(sub)) // max_iter) + 1
        tmp = sub.copy()
        tmp["FunctionID"] = func_id
        out.append(tmp)
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
    
    means = np.array(means)
    stds = np.array(stds)
    ns = np.array(ns)
    tcrit = t_dist.ppf(0.975, df=np.maximum(ns-1, 1))
    se = np.where(ns > 0, stds / np.sqrt(ns), np.nan)
    
    mean_disp_full = apply_llm_smoothing(means, ns, method=smoothing)
    lo_full = mean_disp_full - tcrit * se
    hi_full = mean_disp_full + tcrit * se
    
    sl = slice(x_start - 1, max_iter)
    return mean_disp_full[sl], lo_full[sl], hi_full[sl], ns[sl]

for llm_csv in llm_csvs:
    print(f"\n{'='*60}")
    print(f"Processing: {llm_csv}")
    print('='*60)
    
    model_name = llm_csv.split("/")[-1].replace(".csv", "")

    output_lines = []
    output_lines.append(f"Model: {model_name}\n")
    output_lines.append("="*80 + "\n\n")
    
    # Load and prepare LLM data
    df_llm = pd.read_csv(llm_csv)
    df_llm = df_llm[df_llm["noise"].astype(float).isin(noise_levels)].copy()
    df_llm["Trial"] = df_llm["Trial"].astype(int)
    df_llm = assign_function_ids_by_order(df_llm, max_iter=max_iter)
    
    dimensions = df_llm["dimension"].unique()
    
    for d_dim in dimensions:
        df_dim = df_llm[df_llm["dimension"] == d_dim].copy()
        
        gp_data_dict = {}
        if d_dim in gp_json_paths:
            gp_data_dict = compute_empirical_gp_stats(
                gp_json_paths[d_dim], 
                d_dim, 
                max_iter
            )
        else:
            print(f"No GP JSON path defined for dimension {d_dim}")

        x = np.arange(x_start, max_iter + 1)
        available_kernels = df_dim["kernel"].unique()
        
        for k_name in available_kernels:
            print(f"Plotting {k_name}, dimension={d_dim}, noise={noise} ...")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            gp_key = (k_name, noise)
            if gp_key in gp_data_dict:
                gp_mean_full, gp_lo_full, gp_hi_full = gp_data_dict[gp_key]
                
                sl = slice(x_start - 1, max_iter)
                y_gp = gp_mean_full[sl]
                lo_gp = gp_lo_full[sl]
                hi_gp = gp_hi_full[sl]
                
                ax.plot(x, y_gp, color='#D55E00', lw=3, label="Optimal GP")
                ax.fill_between(x, lo_gp, hi_gp, color='#D55E00', alpha=0.2, linewidth=0)
            else:
                print(f"  [warn] No Empirical GP data for {gp_key}")

            if plot_llm:
                sub = df_dim[
                    (df_dim["kernel"] == k_name) & 
                    (df_dim["noise"].astype(float) == noise)
                ]
                if len(sub) > 0:
                    y_llm, lo_llm, hi_llm, ns = llm_prefix_stats(
                        sub, max_iter=max_iter, smoothing=llm_smoothing
                    )
                    ax.plot(x, y_llm, color="#4285F4", lw=3, label=f"LLM ({llm_smoothing})")
                    ax.fill_between(x, lo_llm, hi_llm, color="#4285F4", alpha=0.20, linewidth=0)
                else:
                    print("  [warn] No LLM rows for this kernel/noise/dim.")
            
            try:
                ell = float(k_name.split("_ls")[-1])
            except:  # noqa: E722
                ell = 1.0
            kernel_type = "SE" if k_name.startswith("SE") else "Matern_05"
            
            R_eff = 200
            means_nn, se_nn = [], []
            for n in range(1, max_iter + 1):
                if n == 1:
                    means_nn.append(0.0)
                    se_nn.append(0.0)
                    continue
                m_abs, v_abs = knn_abs_moments_multidim(
                    n, ell, d_dim, kernel_type=kernel_type, 
                    L=domain_L, sigma_eps2=noise, sigma_f2=1.0
                )
                means_nn.append(m_abs)
                se_nn.append(np.sqrt(v_abs) / np.sqrt(max(R_eff, 1)))
            
            y_nn = np.array(means_nn)[x_start - 1:max_iter]
            se_nn = np.array(se_nn)[x_start - 1:max_iter]
            
            ax.plot(x, y_nn, color='black', lw=3, label="1-NN")
            
            ax.set_xlim(x_start - 0.5, max_iter + 0.5)
            ax.set_ylim(-0.02, 1.20)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.set_xlabel("Trial")
            ax.set_ylabel("MAE")
            
            base_k = "SE" if k_name.startswith("SE") else "Matérn 1/2"
            lam = k_name.split("_ls")[-1]
            ax.set_title(rf"{base_k}, $\lambda$ = {lam}, {d_dim}D, noise = {noise}")
            
            handles = [
                Line2D([0], [0], color='#D55E00', lw=3, label="GP"),
                Line2D([0], [0], color='black', lw=3, label="1-NN"),
            ]
            if plot_llm:
                handles.insert(1, Line2D([0], [0], color="#4285F4", lw=3, label="LLM"))
            
            ax.legend(handles=handles, handlelength=1.0, handletextpad=0.5, loc="upper right")
            
            fig.tight_layout()
            
            out_file = out_dir / (f"{model_name}_LC_{d_dim}D_{llm_smoothing}_{k_name}_noise={noise}_L={int(domain_L)}.png")
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved: {out_file}")

            save_dir = Path("/llm-gaussian-processes/analysis/curves_saved_data")
            save_dir.mkdir(parents=True, exist_ok=True)

            records = []

            for i, t in enumerate(x):
                row = {
                    "model_name": model_name,
                    "dimension": d_dim,
                    "kernel": k_name,
                    "noise": noise,
                    "trial": int(t),
                    "llm_mean": y_llm[i] if plot_llm else np.nan,
                    "llm_lo": lo_llm[i] if plot_llm else np.nan,
                    "llm_hi": hi_llm[i] if plot_llm else np.nan,
                    "gp_mean": y_gp[i] if gp_key in gp_data_dict else np.nan,
                    "gp_lo": lo_gp[i] if gp_key in gp_data_dict else np.nan,
                    "gp_hi": hi_gp[i] if gp_key in gp_data_dict else np.nan,
                    "nn_mean": y_nn[i],
                    "nn_se": se_nn[i],
                }
                records.append(row)

            df_out = pd.DataFrame(records)

            csv_path = save_dir / (
                f"{model_name}_{d_dim}D_{k_name}_noise={noise}_L={int(domain_L)}.csv"
            )

            df_out.to_csv(csv_path, index=False)
            print(f"  Saved data to CSV: {csv_path}")