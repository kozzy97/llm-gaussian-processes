import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from fastcore.script import call_parse

def load_results_from_path(base_path):
    """Load JSON results from a flat directory structure"""
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

def extract_kernel_from_name(model_name):
    """Extract model size (matern12, matern32, matern52, SE) from model name"""
    model_name_lower = model_name.lower()
    if 'matern12' in model_name_lower:
        return 'Matern½ (ls=1)' if 'ls_1' in model_name_lower else 'Matern½ (ls=8)'
    elif 'matern32' in model_name_lower:
        return 'Matern1½ (ls=1)' if 'ls_1' in model_name_lower else 'Matern1½ (ls=8)'
    elif 'matern52' in model_name_lower:
        return 'Matern2½ (ls=1)' if 'ls_1' in model_name_lower else 'Matern2½ (ls=8)'
    elif 'se' in model_name_lower:
        return 'SE (ls=1)' if 'ls_1' in model_name_lower else 'SE (ls=8)'
    else:
        return model_name

def create_ordered_kernel_pairs():
    """Create ordered list of kernel pairs: F order (Matern½, Matern1½, Matern2½, SE) × G order (SE, Matern½)"""
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
    
    color_map = {
        'Matern½ (ls=1)': '#0072B2',
        'Matern½ (ls=8)': '#56B4E9',
        'Matern1½ (ls=1)': '#E69F00',
        'Matern1½ (ls=8)': '#F0C87B',
        'Matern2½ (ls=1)': '#009E73',
        'Matern2½ (ls=8)': '#63C6A7',
        'SE (ls=1)': '#D55E00',
        'SE (ls=8)': '#E99675'
    }
    
    # Get ordered kernel pairs
    kernel_pairs = create_ordered_kernel_pairs()
    
    # Organize data by model size
    model_data = {}
    for model_name, results in results_by_model.items():
        kernel = extract_kernel_from_name(model_name)
        if kernel not in model_data:
            model_data[kernel] = {}
        
        # Calculate average likelihood per prediction for each kernel pair
        for kernel_pair, data in results.items():
            avg_likelihood = data['log_likelihood'] / data['n_predictions'] if data['n_predictions'] > 0 else 0
            model_data[kernel][kernel_pair] = avg_likelihood
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    x = np.arange(len(kernel_pairs))
    width = 0.1
    kernels = ['Matern½ (ls=1)', 'Matern½ (ls=8)', 'Matern1½ (ls=1)', 'Matern1½ (ls=8)',
               'Matern2½ (ls=1)', 'Matern2½ (ls=8)', 'SE (ls=1)', 'SE (ls=8)']
    
    # Plot bars for each model size
    for i, k in enumerate(kernels):
        if k in model_data:
            avg_likelihoods = []
            for kernel_pair in kernel_pairs:
                avg_likelihood = model_data[k].get(kernel_pair, 0)
                avg_likelihoods.append(avg_likelihood)
            
            ax.bar(x + i * width, avg_likelihoods, width, 
                   label=k, color=color_map[k], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Generator/Fitter Kernel Pairs', fontweight='bold')
    ax.set_ylabel('Average Log-Likelihood per Prediction', fontweight='bold')
    ax.set_title('GP Kernel Comparison: Average Log-Likelihood per Prediction by Kernel Pairs', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x + (len(kernels) * width) / 2 - width/2)
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
            kernel = extract_kernel_from_name(model_name)
            if kernel_pair in results:
                data = results[kernel_pair]
                avg_likelihood = data['log_likelihood'] / data['n_predictions'] if data['n_predictions'] > 0 else 0
                print(f"  {kernel}: {data['log_likelihood']:8.1f} total, {avg_likelihood:6.3f} avg ({data['n_predictions']} predictions)")

@call_parse
def main(base_path: str = None, save_path: str = None) -> None:
    results_by_model = load_results_from_path(base_path)
    
    if not results_by_model:
        print("No results found! Check your base path.")
        return
    
    print(f"\nFound results for {len(results_by_model)} models:")
    for model_name in sorted(results_by_model.keys()):
        kernel = extract_kernel_from_name(model_name)
        print(f"  - {model_name} ({kernel})")
    
    # Create plot
    fig = create_model_comparison_plot(results_by_model, save_path)
    
    # Print summary
    print_summary(results_by_model)


# USAGE:
# python plot_likelihoods_gp.py --base_path /llm-gaussian-processes/analysis/saved_data/priors_likelihood_analysis/unadjusted/GPs --save_path /llm-gaussian-processes/analysis/plots/priors/raw/camera-ready/GP_average_likelihoods.png

