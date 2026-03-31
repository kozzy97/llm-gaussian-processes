import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from fastcore.script import call_parse

def load_results_from_path(base_path):
    results_by_model = {}
    base_path = Path(base_path)
    
    json_files = list(base_path.glob("*_gp_kernel_comparison.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_name = json_file.stem.replace('_gp_kernel_comparison', '')
            results_by_model[model_name] = data
            print(f"Loaded results for {model_name}")
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results_by_model

def extract_model_size_from_name(model_name):
    model_name_lower = model_name.lower()
    if '8b' in model_name_lower:
        return '8B'
    elif '14b' in model_name_lower:
        return '14B'
    elif '32b' in model_name_lower:
        return '32B'
    else:
        return model_name

def create_ordered_kernel_pairs():
    fitter_order = ['Matern½', 'Matern1½', 'Matern2½', 'SE']
    generator_order = ['SE', 'Matern½']
    
    ordered_pairs = []
    for f in fitter_order:
        for g in generator_order:
            pair_key = f"G: {g}\nF: {f}"
            ordered_pairs.append(pair_key)
    
    return ordered_pairs

def create_model_comparison_plot(results_by_model, save_path=None):
    """Create grouped bar plot with custom ordering and no value labels"""
    
    # Color mapping
    color_map = {
        '8B': "#0072B2",
        '14B': "#D55E00",
        '32B': "#009E73"
    }
    
    kernel_pairs = create_ordered_kernel_pairs()
    
    # Organize data by model size
    model_data = {}
    for model_name, results in results_by_model.items():
        model_size = extract_model_size_from_name(model_name)
        if model_size not in model_data:
            model_data[model_size] = {}
        
        # Calculate average likelihood per prediction for each kernel pair
        for kernel_pair, data in results.items():
            avg_likelihood = data['log_likelihood'] / data['n_predictions'] if data['n_predictions'] > 0 else 0
            model_data[model_size][kernel_pair] = avg_likelihood
    
    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Bar positions
    x = np.arange(len(kernel_pairs))
    width = 0.25
    model_sizes = ['8B', '14B', '32B']
    
    # Plot bars for each model size
    for i, model_size in enumerate(model_sizes):
        if model_size in model_data:
            avg_likelihoods = []
            for kernel_pair in kernel_pairs:
                avg_likelihood = model_data[model_size].get(kernel_pair, 0)
                avg_likelihoods.append(avg_likelihood)
            
            ax.bar(x + i * width, avg_likelihoods, width, 
                   label=model_size, color=color_map[model_size], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Generator/Fitter Kernel Pairs', fontweight='bold')
    ax.set_ylabel('Average Log-Likelihood per Prediction', fontweight='bold')
    ax.set_title('Model Size Comparison: Average Log-Likelihood per Prediction by Kernel Pairs', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(kernel_pairs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=0.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig

def print_summary(results_by_model):
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    kernel_pairs = create_ordered_kernel_pairs()
    
    for kernel_pair in kernel_pairs:
        print(f"\n{kernel_pair}:")
        print("-" * 50)
        
        for model_name, results in sorted(results_by_model.items()):
            model_size = extract_model_size_from_name(model_name)
            if kernel_pair in results:
                data = results[kernel_pair]
                avg_likelihood = data['log_likelihood'] / data['n_predictions'] if data['n_predictions'] > 0 else 0
                print(f"  {model_size:3s}: {data['log_likelihood']:8.1f} total, {avg_likelihood:6.3f} avg ({data['n_predictions']} predictions)")

@call_parse
def main(base_path: str = None, save_path: str = None) -> None:
    results_by_model = load_results_from_path(base_path)
    
    if not results_by_model:
        print("No results found! Check your base path.")
        return
    
    print(f"\nFound results for {len(results_by_model)} models:")
    for model_name in sorted(results_by_model.keys()):
        model_size = extract_model_size_from_name(model_name)
        print(f"  - {model_name} ({model_size})")
    
    fig = create_model_comparison_plot(results_by_model, save_path)
    
    print_summary(results_by_model)


# USAGE:
# python plot_likelihoods_base.py --base_path /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/BASE --save_path /llm-gaussian-processes/analysis/plots/priors/raw/camera-ready/BASE_average_likelihoods.png
# python plot_likelihoods_base.py --base_path /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/tau2-adjusted/BASE --save_path /llm-gaussian-processes/analysis/plots/priors/raw/camera-ready/BASE_average_likelihoods_tau2.png

