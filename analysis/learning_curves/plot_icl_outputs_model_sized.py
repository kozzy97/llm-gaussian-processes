from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as t_dist
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

icl_files = {
    "8B Base":  "/llm-gaussian-processes/model_prediction_data/BASE/Qwen3-8B-unsloth-bnb-4bit.csv",
    "14B Base": "/llm-gaussian-processes/model_prediction_data/BASE/Qwen3-14B-unsloth-bnb-4bit.csv",
    "32B Base": "/llm-gaussian-processes/model_prediction_data/BASE/Qwen3-32B-unsloth-bnb-4bit.csv",
    # "8B FT":    "/path/to/FT/Qwen3-8B-FT.csv",
    # "14B FT":   "/path/to/FT/Qwen3-14B-FT.csv",
    # "32B FT":   "/path/to/FT/Qwen3-32B-FT.csv",
}

out_dir = Path("/llm-gaussian-processes/analysis/plots/learning_curves_LC_UC")
out_dir.mkdir(parents=True, exist_ok=True)

noise_filter = 0.001
max_iter     = 50
x_start      = 2
ylim_top     = 0.40
llm_smoothing = "none" # "isotonic" | "ema" | "none"

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
    blocks = [[0]]
    for i in range(1, len(x)):
        blocks.append([i])
        while len(blocks) >= 2:
            last = blocks[-1]
            prev = blocks[-2]
            m_last = np.average(x[last], weights=w[last])
            m_prev = np.average(x[prev], weights=w[prev])
            if m_prev > m_last + 1e-15:
                # pool
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

def read_llm_files(icl_files, noise_filter=0.001):
    frames = []
    for label, path in icl_files.items():
        df = pd.read_csv(path)
        for col in ["Trial", "Absolute_Error", "kernel", "noise"]:
            if col not in df.columns:
                raise ValueError(f"'{col}' column missing in {path}")
        df = df.copy()
        df["model"] = label
        df["Trial"] = df["Trial"].astype(int)
        df = df[df["noise"].astype(float) == float(noise_filter)]
        df = df[df["Trial"].between(1, max_iter)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def assign_function_ids_by_order(df, max_iter=50):
    df = df.copy()
    df["__row__"] = np.arange(len(df))
    out = []
    for _, sub in df.groupby(["kernel", "noise", "model"], sort=False):
        sub = sub.sort_values("__row__")  # preserve file order

        func_id = (np.arange(len(sub)) // max_iter) + 1

        ok = True
        tmp = sub.copy()
        tmp["FunctionID"] = func_id
        sizes = tmp.groupby("FunctionID")["Trial"].nunique()
        if sizes.empty or sizes.min() < max_iter or sizes.max() > max_iter:
            ok = False

        if not ok:
            starts = (sub["Trial"] == 1).astype(int).values
            if starts.size == 0:
                continue
            if starts[0] == 0: starts[0] = 1
            func_id = np.cumsum(starts)

        sub = sub.copy()
        sub["FunctionID"] = func_id
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
    dfs   = np.maximum(ns - 1, 1)
    tcrit = t_dist.ppf(0.975, df=dfs)
    se    = np.where(ns > 0, stds / np.sqrt(ns), np.nan)

    mean_disp_full = apply_llm_smoothing(means, ns, method=smoothing)

    lo_full = mean_disp_full - tcrit * se
    hi_full = mean_disp_full + tcrit * se

    sl = slice(x_start-1, max_iter)
    return mean_disp_full[sl], lo_full[sl], hi_full[sl], ns[sl]

df = read_llm_files(icl_files, noise_filter=noise_filter)
df = assign_function_ids_by_order(df, max_iter=max_iter)

kernels_present = df["kernel"].astype(str).unique().tolist()
kernels_present.sort()

model_labels = list(icl_files.keys())
palette = plt.get_cmap("tab10")
COLORS = ["#4285F4", "#EA4335", "#0F9D58"]
model_colors = {lbl: COLORS[i % len(COLORS)] for i, lbl in enumerate(model_labels)}

for k_name in kernels_present:
    print(f"Plotting LLMs for kernel={k_name}, noise={noise_filter} ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(x_start, max_iter + 1)

    any_plotted = False
    for lbl in model_labels:
        sub = df[(df["model"] == lbl) &
                 (df["kernel"] == k_name) &
                 (df["noise"].astype(float) == float(noise_filter))]

        if len(sub) == 0:
            continue

        y, lo, hi, ns = llm_prefix_stats(sub, max_iter=max_iter, smoothing=llm_smoothing)
        ax.plot(x, y, color=model_colors[lbl], lw=3, label=lbl)
        ax.fill_between(x, lo, hi, color=model_colors[lbl], alpha=0.20, linewidth=0)
        any_plotted = True
        print(f"  {lbl}: prefix-complete functions used (max): ~{int(np.nanmax(ns))}")

    if not any_plotted:
        print("  [warn] No data for this kernel; skipping figure.")
        plt.close(fig)
        continue

    ax.set_xlim(x_start - 0.5, max_iter + 0.5)
    ax.set_ylim(-0.02, ylim_top)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel("Trial")
    ax.set_ylabel("MAE")
    try:
        base_k = "SE" if k_name.startswith("SE") else "Matérn 1/2"
        lam    = k_name.split("_ls")[-1]
        title  = rf"{base_k}, $\lambda$ = {lam}, noise = {noise_filter}"
    except Exception:
        title  = f"kernel = {k_name}, noise = {noise_filter}"
    ax.set_title(title)

    handles = [Line2D([0], [0], color=model_colors[lbl], lw=3, label=lbl) for lbl in model_labels if lbl in df["model"].unique()]
    ax.legend(handles=handles, handlelength=1.0, handletextpad=0.5, loc="upper right")

    fig.tight_layout()
    out_file = out_dir / f"LLM_only_{llm_smoothing}_{k_name}_noise={noise_filter}.png"
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved: {out_file}")
