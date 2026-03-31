import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastcore.script import call_parse
import GPy
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def load_gp_data(folder_path):
    output_dict = {}
    lengthscales = [1, 8]
    kernels = ["matern12", "matern32", "matern52", "SE"]
    for k in kernels:
        for l in lengthscales:
            file_path = f"{folder_path}/gp_predictions_{k}_ls_{l}_gp_kernel_comparison.json"
            with open(file_path, 'r') as f:
                data = json.load(f)
            output_dict[f"{k}_l{l}"] = data
    return output_dict

gp_data = load_gp_data("/gp-icl/icl/outputs/GPs")

def load_model_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def normalize_data(model_data, gp_data, gp_comparison_lengthscale):
    output_dict = {
        'G: SE\nF: Matern½': model_data['G: SE\nF: Matern½']['log_likelihood'] - gp_data[f"matern12_l{gp_comparison_lengthscale}"]['G: SE\nF: Matern½']['log_likelihood'],
        'G: Matern½\nF: Matern½': model_data['G: Matern½\nF: Matern½']['log_likelihood'] - gp_data[f"matern12_l{gp_comparison_lengthscale}"]['G: Matern½\nF: Matern½']['log_likelihood'],
        'G: SE\nF: Matern1½': model_data['G: SE\nF: Matern1½']['log_likelihood'] - gp_data[f"matern32_l{gp_comparison_lengthscale}"]['G: SE\nF: Matern1½']['log_likelihood'],
        'G: Matern½\nF: Matern1½': model_data['G: Matern½\nF: Matern1½']['log_likelihood'] - gp_data[f"matern32_l{gp_comparison_lengthscale}"]['G: Matern½\nF: Matern1½']['log_likelihood'],
        'G: SE\nF: Matern2½': model_data['G: SE\nF: Matern2½']['log_likelihood'] - gp_data[f"matern52_l{gp_comparison_lengthscale}"]['G: SE\nF: Matern2½']['log_likelihood'],
        'G: Matern½\nF: Matern2½': model_data['G: Matern½\nF: Matern2½']['log_likelihood'] - gp_data[f"matern52_l{gp_comparison_lengthscale}"]['G: Matern½\nF: Matern2½']['log_likelihood'],
        'G: SE\nF: SE': model_data['G: SE\nF: SE']['log_likelihood'] - gp_data[f"SE_l{gp_comparison_lengthscale}"]['G: SE\nF: SE']['log_likelihood'],
        'G: Matern½\nF: SE': model_data['G: Matern½\nF: SE']['log_likelihood'] - gp_data[f"SE_l{gp_comparison_lengthscale}"]['G: Matern½\nF: SE']['log_likelihood'],
    }
    return output_dict

def plot_normalized_data(normalized_data, model_name):
    df = pd.Series(normalized_data, name='Δ log ℒ').to_frame()
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df['Δ log ℒ'])
    plt.axhline(0, linestyle='--', label='Zero Line')
    plt.yscale('symlog', linthresh=1e4)   # logarithmic outside ±10 000
    plt.ylabel('Log-likelihood difference (symlog)')
    plt.xticks(rotation=45, ha='right')
    plt.title('Normalized Log-likelihood Differences')
    plt.legend()
    plt.tight_layout()
    out_path = f"/gp-icl/analysis/normalized_plots/normalized_log_likelihood_diff_{model_name}.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

model_data = load_model_data("/gp-icl/icl/outputs/BASE/Qwen3-8B-unsloth-bnb-4bit_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 1)
plot_normalized_data(normalized_data, "BASE_Qwen3-8B_GPls=1")

model_data = load_model_data("/gp-icl/icl/outputs/GRPO/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-5000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 1)
plot_normalized_data(normalized_data, "GRPO_Qwen3-8B_GPls=1")

model_data = load_model_data("/gp-icl/icl/outputs/SFT/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-5000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 1)
plot_normalized_data(normalized_data, "SFT_Qwen3-8B_GPls=1")

model_data = load_model_data("/gp-icl/icl/outputs/BASE/Qwen3-8B-unsloth-bnb-4bit_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 8)
plot_normalized_data(normalized_data, "BASE_Qwen3-8B_GPls=8")

model_data = load_model_data("/gp-icl/icl/outputs/GRPO/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-5000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 8)
plot_normalized_data(normalized_data, "GRPO_Qwen3-8B_GPls=8")

model_data = load_model_data("/gp-icl/icl/outputs/SFT/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-5000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 8)
plot_normalized_data(normalized_data, "SFT_Qwen3-8B_GPls=8")

model_data = load_model_data("/gp-icl/icl/outputs/SFT/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-10000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 8)
plot_normalized_data(normalized_data, "SFT_10k_Qwen3-8B_GPls=8")

model_data = load_model_data("/gp-icl/icl/outputs/SFT/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-15000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 8)
plot_normalized_data(normalized_data, "SFT_15k_Qwen3-8B_GPls=8")

model_data = load_model_data("/gp-icl/icl/outputs/SFT/Qwen3-8B-unsloth-bnb-4bit-ls-1-mu-0.0-sd-0.01-checkpoint-20000_gp_kernel_comparison.json")
normalized_data = normalize_data(model_data, gp_data, 8)
plot_normalized_data(normalized_data, "SFT_20k_Qwen3-8B_GPls=8")
