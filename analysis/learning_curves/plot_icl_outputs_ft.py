from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as t_dist
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

parameter_size = "8B"  # for naming only
overall_algorithm = "SFT"  # for naming only
reward_type = "matern"  # for naming only
ls = 1
model_family = "Qwen3"  # for naming only
quantization = "unsloth-bnb-4bit"  # for naming only

icl_files = {
    f"{parameter_size} Base":  f"/llm-gaussian-processes/model_prediction_data/BASE/{model_family}-{parameter_size}-{quantization}.csv",
    "Ckpt 5k": f"/llm-gaussian-processes/model_prediction_data/{overall_algorithm}/{reward_type}/{model_family}-{parameter_size}-{quantization}-ls-{ls}-mu-0.0-sd-0.01-checkpoint-5000.csv",
    "Ckpt 10k": f"/llm-gaussian-processes/model_prediction_data/{overall_algorithm}/{reward_type}/{model_family}-{parameter_size}-{quantization}-ls-{ls}-mu-0.0-sd-0.01-checkpoint-10000.csv"
}

out_dir = Path("/llm-gaussian-processes/analysis/plots/learning_curves_ft_checkpoints")
out_dir.mkdir(parents=True, exist_ok=True)

noise_filter   = 0.001
max_iter       = 50
x_start        = 2
ylim_top       = 0.40
llm_smoothing  = "none" # "isotonic" | "ema" | "none"
fallback_global_trial_mean = True  # set False to STRICT same-kernel/model/noise only

plt.rcParams.update({
    'font.size': 26, 'axes.titlesize': 24, 'axes.labelsize': 26,
    'legend.fontsize': 22, 'xtick.labelsize': 26, 'ytick.labelsize': 26
})

def smooth_causal_ema(y, span=10):
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
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    x = -y
    blocks = [[0]]
    for i in range(1, len(x)):
        blocks.append([i])
        while len(blocks) >= 2:
            last, prev = blocks[-1], blocks[-2]
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

def apply_llm_smoothing(means, ns, method="none"):
    if method == "isotonic":
        return isotonic_nonincreasing(means, w=np.where(np.isnan(ns), 0.0, ns))
    elif method == "ema":
        return smooth_causal_ema(means, span=10)
    else:
        return means

def read_llm_files(icl_files, noise_filter=0.001):
    """Read CSVs, attach 'model', filter by noise & trial window."""
    required_cols = [
        "Trial", "True_Y", "LLM_Prediction", "Absolute_Error",
        "seed", "kernel", "noise"
    ]
    frames = []
    for label, path in icl_files.items():
        df = pd.read_csv(path)
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"'{col}' column missing in {path}")
        df = df.copy()
        df["model"] = label
        df["Trial"] = df["Trial"].astype(int)
        df = df[df["noise"].astype(float) == float(noise_filter)]
        df = df[df["Trial"].between(1, max_iter)]
        frames.append(df)
    if not frames:
        raise RuntimeError("No data loaded. Check file paths and noise filter.")
    return pd.concat(frames, ignore_index=True)

def assign_function_ids_by_order(df, max_iter=50):
    df = df.copy()
    df["__row__"] = np.arange(len(df))
    out = []
    for _, sub in df.groupby(["kernel", "noise", "model"], sort=False):
        sub = sub.sort_values("__row__")
        func_id = (np.arange(len(sub)) // max_iter) + 1
        # validate
        tmp = sub.copy(); tmp["FunctionID"] = func_id
        sizes = tmp.groupby("FunctionID")["Trial"].nunique()
        ok = not (sizes.empty or sizes.min() < max_iter or sizes.max() > max_iter)
        if not ok:
            starts = (sub["Trial"] == 1).astype(int).values
            if starts.size == 0:
                continue
            if starts[0] == 0:
                starts[0] = 1
            func_id = np.cumsum(starts)
        sub = sub.copy()
        sub["FunctionID"] = func_id
        out.append(sub)
    out = pd.concat(out, ignore_index=True)
    out.drop(columns="__row__", inplace=True)
    return out

def materialize_full_trial_grid(df, max_iter=50):
    df = df.copy()
    keys = ["kernel", "model", "noise", "FunctionID"]
    full = []
    for _, sub in df.groupby(keys, sort=False):
        sub = sub.set_index("Trial").sort_index()
        # reindex to full 1..max_iter
        sub = sub.reindex(range(1, max_iter + 1))
        sub[keys] = sub[keys].ffill().bfill()  # carry group keys
        full.append(sub.reset_index().rename(columns={"index": "Trial"}))
    return pd.concat(full, ignore_index=True)

def impute_abs_error_within_bucket_then_global(df, fallback_global=True):
    df = df.copy()
    df["Trial"] = df["Trial"].astype(int)

    bucket_mean = (df.groupby(["kernel", "model", "noise", "Trial"])["Absolute_Error"]
                     .transform(lambda s: pd.to_numeric(s, errors="coerce").mean(skipna=True)))

    need1 = df["Absolute_Error"].isna() & bucket_mean.notna()
    df.loc[need1, "Absolute_Error"] = bucket_mean[need1]

    filled1 = int(need1.sum())

    if fallback_global:
        trial_means = (df.groupby("Trial")["Absolute_Error"]
                         .apply(lambda s: pd.to_numeric(s, errors="coerce").mean(skipna=True)))
        mapped = df["Trial"].map(trial_means)
        need2 = df["Absolute_Error"].isna() & mapped.notna()
        df.loc[need2, "Absolute_Error"] = mapped[need2]
        filled2 = int(need2.sum())
    else:
        filled2 = 0

    remaining = int(df["Absolute_Error"].isna().sum())
    print(f"[impute] Filled AE: bucket={filled1}, globalTrial={filled2}, remaining={remaining}")
    if remaining > 0:
        print("WARNING: Some AE remain NaN because fallback is disabled and bucket had no data.")

    return df

def llm_prefix_stats(df_model_kernel, max_iter=50, smoothing="none"):
    pivot = df_model_kernel.pivot(index="FunctionID", columns="Trial", values="Absolute_Error")
    pivot = pivot[pivot.columns.intersection(range(1, max_iter + 1))]

    means, stds, ns = [], [], []
    for t in range(1, max_iter + 1):
        ok_rows = pivot.loc[:, 1:t].notna().all(axis=1)
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

    sl = slice(x_start - 1, max_iter)
    return mean_disp_full[sl], lo_full[sl], hi_full[sl], ns[sl]

df = read_llm_files(icl_files, noise_filter=noise_filter)
df = assign_function_ids_by_order(df, max_iter=max_iter)
df_full = materialize_full_trial_grid(df, max_iter=max_iter)
df_aug = impute_abs_error_within_bucket_then_global(df_full, fallback_global=fallback_global_trial_mean)
kernels_present = df_aug["kernel"].astype(str).unique().tolist()
kernels_present.sort()

model_labels = list(icl_files.keys())
COLORS = ["#4285F4", "#EA4335", "#0F9D58"]
model_colors = {lbl: COLORS[i % len(COLORS)] for i, lbl in enumerate(model_labels)}

for k_name in kernels_present:
    print(f"Plotting LLMs for kernel={k_name}, noise={noise_filter} ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(x_start, max_iter + 1)

    any_plotted = False
    for lbl in model_labels:
        sub = df_aug[(df_aug["model"] == lbl) &
                     (df_aug["kernel"].astype(str) == str(k_name)) &
                     (df_aug["noise"].astype(float) == float(noise_filter))]
        if len(sub) == 0:
            continue

        y, lo, hi, ns = llm_prefix_stats(sub, max_iter=max_iter, smoothing=llm_smoothing)
        ax.plot(x, y, color=model_colors[lbl], lw=3, label=lbl)
        ax.fill_between(x, lo, hi, color=model_colors[lbl], alpha=0.20, linewidth=0)
        any_plotted = True
        print(f"  {lbl}: prefix-complete functions at n={max_iter}: {int(ns[-1]) if len(ns) else 0}")

    save_dir = Path(
        "/llm-gaussian-processes/analysis/curves_saved_data"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for lbl in model_labels:
        sub = df_aug[(df_aug["model"] == lbl) &
                     (df_aug["kernel"].astype(str) == str(k_name)) &
                     (df_aug["noise"].astype(float) == float(noise_filter))]
        if len(sub) == 0:
            continue

        y, lo, hi, ns = llm_prefix_stats(
            sub, max_iter=max_iter, smoothing=llm_smoothing
        )

        for i, t in enumerate(x):
            records.append({
                "model": lbl,
                "kernel": k_name,
                "noise": noise_filter,
                "trial": int(t),
                "mean": float(y[i]),
                "lo": float(lo[i]),
                "hi": float(hi[i]),
                "n_prefix_complete": int(ns[i]),
            })

    df_curve = pd.DataFrame(records)

    csv_path = save_dir / (
        f"checkpoints_{overall_algorithm}_{reward_type}_ls-{ls}_{model_family}-{parameter_size}-{quantization}_{k_name}_noise={noise_filter}.csv"
    )
    df_curve.to_csv(csv_path, index=False)
    print(f"  → Saved LC data CSV: {csv_path}")


    if not any_plotted:
        print("[Warning] No data for this kernel, skipping.")
        plt.close(fig)
        continue

    ax.set_xlim(x_start - 0.5, max_iter + 0.5)
    ax.set_ylim(-0.02, ylim_top)
    ax.set_ylim(0, 0.40)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel("Trial")
    ax.set_ylabel("MAE")
    try:
        base_k = "SE" if str(k_name).startswith("SE") else "Matérn 1/2"
        lam    = str(k_name).split("_ls")[-1]
        title  = rf"{base_k}, $\lambda$ = {lam}, noise = {noise_filter}"
    except Exception:
        title  = f"{k_name}, noise = {noise_filter}"
    ax.set_title(title)

    handles = [Line2D([0], [0], color=model_colors[lbl], lw=3, label=lbl)
               for lbl in model_labels if lbl in df_aug["model"].unique()]
    ax.legend(handles=handles, handlelength=1.0, handletextpad=0.5, loc="upper right")

    fig.tight_layout()
    out_file = out_dir / f"{model_family}-{parameter_size}-{quantization}_{overall_algorithm}_{reward_type}_ls-{8}_{k_name}_noise={noise_filter}.png"
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved: {out_file}")
