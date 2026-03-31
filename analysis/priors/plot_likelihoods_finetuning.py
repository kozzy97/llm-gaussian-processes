import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from fastcore.script import call_parse


model_family = "Qwen3"
train_type = "log_matern"

def create_ordered_kernel_pairs():
    """F order (Matern½, Matern1½, Matern2½, SE) × G order (SE, Matern½)."""
    fitter_order = ['Matern½', 'Matern1½', 'Matern2½', 'SE']
    generator_order = ['SE', 'Matern½']
    return [f"G: {g}\nF: {f}" for f in fitter_order for g in generator_order]

def avg_ll_per_pred(entry):
    n = entry.get('n_predictions', 0)
    return (entry.get('log_likelihood', 0.0) / n) if n and n > 0 else 0.0

def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

def load_base_map(base_file: Path):
    """
    Load the base JSON file and return {kernel_pair -> avg_ll_per_pred}.
    """
    if base_file is None or not base_file.exists():
        print(f"ERROR: Base file not found: {base_file}")
        return {}
    data = load_json(base_file)
    return {k: avg_ll_per_pred(v) for k, v in data.items()}

def collect_method_results(method_dir: Path, model_size: str, ls: int, checkpoints: list[int]):
    out = {}
    if method_dir is None or not method_dir.exists():
        print(f"WARNING: Method directory not found: {method_dir}")
        return out
    for ckpt in checkpoints:
        pattern = f"{model_family}*{model_size}*ls-{ls}*checkpoint-{ckpt}_gp_kernel_comparison.json"
        files = list(method_dir.glob(pattern))
        ckpt_agg = {}
        ckpt_counts = {}
        for p in files:
            try:
                data = load_json(p)
                for kpair, entry in data.items():
                    val = avg_ll_per_pred(entry)
                    ckpt_agg[kpair] = ckpt_agg.get(kpair, 0.0) + val
                    ckpt_counts[kpair] = ckpt_counts.get(kpair, 0) + 1
            except Exception as e:
                print(f"Error loading {method_dir.name} {p}: {e}")
        # Average across duplicates if multiple files matched
        out[ckpt] = {k: (ckpt_agg[k] / ckpt_counts[k]) for k in ckpt_agg} if ckpt_agg else {}
    return out

def make_color_shades(base_hex_list, n):
    if n <= len(base_hex_list):
        idxs = np.linspace(0, len(base_hex_list)-1, n).round().astype(int).tolist()
        return [base_hex_list[i] for i in idxs]
    else:
        return base_hex_list + [base_hex_list[-1]]*(n - len(base_hex_list))

def plot_diff_for_lengthscale(
    diffs_sft, diffs_grpo, kernel_pairs, checkpoints, model_size: str, ls: int,
    save_path: Path | None = None
):
    x = np.arange(len(kernel_pairs))
    n_series = len(checkpoints)*2  # SFT + GRPO
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    # Palettes (SFT = dark blues, GRPO = oranges)
    blues = ["#0A1931", "#102A43", "#1F3B5C", "#294766", "#365B7D", "#3F6890", "#4875A3"]
    oranges = ["#7B1F00", "#993300", "#BF4C00", "#D65F00", "#E57300", "#F08519", "#FFA041"]

    sft_colors = make_color_shades(blues, len(checkpoints))
    grpo_colors = make_color_shades(oranges, len(checkpoints))

    offset = -0.5 * n_series * width + 0.5 * width

    # SFT bars
    for i, ckpt in enumerate(checkpoints):
        series = [diffs_sft.get(ckpt, {}).get(kp, np.nan) for kp in kernel_pairs]
        ax.bar(x + offset + i*width, series, width, label=f"SFT {ckpt//1000}k", color=sft_colors[i], alpha=0.9)

    # GRPO bars
    base_idx = len(checkpoints)
    for j, ckpt in enumerate(checkpoints):
        series = [diffs_grpo.get(ckpt, {}).get(kp, np.nan) for kp in kernel_pairs]
        ax.bar(x + offset + (base_idx + j)*width, series, width, label=f"GRPO {ckpt//1000}k", color=grpo_colors[j], alpha=0.9)

    ax.set_title(f"{model_family} - {model_size} — ls-{ls}: Δ Avg Log-Likelihood per Prediction vs Base", fontweight='bold')
    ax.set_ylabel("Δ Average Log-Likelihood per Prediction (model − base)", fontweight='bold')
    ax.set_xlabel("Generator / Fitter Kernel Pairs", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(kernel_pairs, rotation=45, ha='right')
    ax.axhline(0.0, lw=1.0, color='#444444', alpha=0.7)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(ncols=2, frameon=False)
    ax.set_yscale('symlog', linthresh=0.1)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig

def compute_diffs_vs_base(base_map, method_map, kernel_pairs):
    diffs = {}
    for ckpt, kp_map in method_map.items():
        diffs[ckpt] = {}
        for kp in kernel_pairs:
            base_val = base_map.get(kp, np.nan)
            method_val = kp_map.get(kp, np.nan)
            diffs[ckpt][kp] = (method_val - base_val) if not (np.isnan(base_val) or np.isnan(method_val)) else np.nan
    return diffs

def print_diff_summary(diffs, label, kernel_pairs):
    print("\n" + "="*80)
    print(f"{label}  Δ Avg Log-Likelihood per Prediction (model − base)")
    print("="*80)
    for kp in kernel_pairs:
        row = [f"{ckpt//1000}k:{diffs.get(ckpt, {}).get(kp, np.nan): .4f}" for ckpt in sorted(diffs.keys())]
        print(f"{kp:28s}  " + "  ".join(row))

@call_parse
def main(
    base_file: str=None,  # Path to the single base JSON file (no checkpoints; used for all ls)
    grpo_path: str=None,  # Directory with GRPO JSONs
    sft_path: str=None,   # Directory with SFT JSONs
    model_size: str='8B',  # e.g., 8B/14B/32B (used in filename pattern matching for GRPO/SFT)
    checkpoints: str='5000,10000,15000,20000',  # Comma-separated integers
    lengthscales: str='1,8',  # Comma-separated ints; we will plot each separately
    save_dir: str=None  # Optional directory to save figures
):
    base_file_p = Path(base_file) if base_file else None
    grpo_dir = Path(grpo_path) if grpo_path else None
    sft_dir  = Path(sft_path) if sft_path else None
    ckpts = [int(x.strip()) for x in checkpoints.split(',') if x.strip()]
    ls_list = [int(x.strip()) for x in lengthscales.split(',') if x.strip()]
    kernel_pairs = create_ordered_kernel_pairs()
    save_dir_p = Path(save_dir) if save_dir else None

    print(f"Base file: {base_file_p}")
    print(f"Model size: {model_size}")
    print(f"Checkpoints: {ckpts}")
    print(f"Lengthscales: {ls_list}")
    print(f"Directories:\n  GRPO: {grpo_dir}\n  SFT : {sft_dir}")

    base_map = load_base_map(base_file_p)
    if not base_map:
        print("WARNING: No usable base data loaded. All diffs will be NaN where base is missing.")

    for ls in ls_list:
        print(f"\nProcessing ls-{ls} ...")
        grpo_map = collect_method_results(grpo_dir, model_size=model_size, ls=ls, checkpoints=ckpts)
        sft_map  = collect_method_results(sft_dir,  model_size=model_size, ls=ls, checkpoints=ckpts)

        diffs_grpo = compute_diffs_vs_base(base_map, grpo_map, kernel_pairs)
        diffs_sft  = compute_diffs_vs_base(base_map, sft_map,  kernel_pairs)

        # Print concise summaries
        print_diff_summary(diffs_sft,  f"SFT  ({model_size}, ls-{ls})", kernel_pairs)
        print_diff_summary(diffs_grpo, f"GRPO ({model_size}, ls-{ls})", kernel_pairs)

        # Plot
        save_path = None
        if save_dir_p:
            save_dir_p.mkdir(parents=True, exist_ok=True)
            grpo_reward_func = "abs" if "absolute" in grpo_path else "log"
            save_path = save_dir_p / f"{model_family}_{train_type}_{model_size}_ls-{ls}_rew-{grpo_reward_func}_delta_avg_ll_per_pred.png"

        plot_diff_for_lengthscale(
            diffs_sft=diffs_sft,
            diffs_grpo=diffs_grpo,
            kernel_pairs=kernel_pairs,
            checkpoints=ckpts,
            model_size=model_size,
            ls=ls,
            save_path=save_path
        )

"""
USAGE EXAMPLE:

python plot_likelihoods_finetuning.py \
  --base_file /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/BASE/Qwen3-8B-unsloth-bnb-4bit_gp_kernel_comparison.json \
  --grpo_path /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/GRPO/absolute \
  --sft_path  /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/SFT \
  --model_size 8B \
  --checkpoints 1000,2000,5000,10000 \
  --lengthscales 1,8 \
  --save_dir /llm-gaussian-processes/analysis/plots/priors/raw/camera-ready

Notes:
- The single base JSON is used for *all* lengthscales (ls-1, ls-8), since ls refers to the finetuning data, not the base.
- The script averages across files if multiple JSONs match a given (method, ls, checkpoint).
- Δ is computed per kernel pair as (method avg − base avg).
- One figure per lengthscale (ls-1 and ls-8), with SFT (blues) and GRPO (oranges).
"""

""" Llama
python plot_likelihoods_finetuning.py \
  --base_file /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/BASE/Llama-3.1-8B-Instruct-unsloth-bnb-4bit_gp_kernel_comparison.json \
  --grpo_path /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/GRPO/absolute \
  --sft_path  /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/SFT/squared_exponential \
  --model_size 8B \
  --checkpoints 1000,2000,5000,10000 \
  --lengthscales 1,8 \
  --save_dir /llm-gaussian-processes/analysis/plots/priors/raw/camera-ready
"""

""" Qwen Matern
python plot_likelihoods_finetuning.py \
  --base_file /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/BASE/Qwen3-8B-unsloth-bnb-4bit_gp_kernel_comparison.json \
  --grpo_path /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/GRPO/log_prob_matern \
  --sft_path  /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/SFT/matern \
  --model_size 8B \
  --checkpoints 1000,2000,5000,10000 \
  --lengthscales 1,8 \
  --save_dir /llm-gaussian-processes/analysis/plots/priors/raw/camera-ready

"""