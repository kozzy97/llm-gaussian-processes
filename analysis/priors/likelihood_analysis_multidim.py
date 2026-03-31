import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastcore.script import call_parse
import GPy
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file):
    """Load CSV and prepare data for GP analysis"""
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    max_trials = df['Trial'].max()
    num_rows = len(df)
    df['function_id'] = [i for i in range((num_rows//max_trials)) for _ in range(1, max_trials+1)]
    return df

def parse_multidim_inputs(df):
    def parse_vec(s):
        parts = s.split(',')
        coords = []
        for p in parts:
            _, val = p.split(':')
            coords.append(float(val.strip()))
        return np.array(coords)

    df['True_X_vec'] = df['True_X_Str'].apply(parse_vec)
    df['dim'] = df['True_X_vec'].apply(lambda x: len(x))
    return df


def fit_gpy_and_calculate_likelihood(df, kernel_generator, kernel_fitter, fitter_length_scale: int, noise: bool, verbose: bool = False):
    k_generators = ['SE_ls1', 'SE_ls8'] if kernel_generator.startswith('SE') else ['Matern_05_ls1', 'Matern_05_ls8']
    df = df[df['kernel'].isin(k_generators)]  # Filter by kernel type used to generate the data
    
    noise_val = df['noise'].unique()[0] if not df['noise'].isnull().all() and noise else 0.0
    function_groups = df.groupby(['seed', 'kernel', 'noise', 'function_id'])
    total_log_likelihood = 0.0
    n_predictions = 0
    for group_id, group_data in function_groups:
        # Sort by trial number to ensure proper sequence
        group_data = group_data.sort_values('Trial')
        # For each trial, use previous trials to predict current Y
        for i in range(1, len(group_data)):  # Start from trial 2
            if verbose:
                print(f"Processing group {group_id}, trial {i+1}/{len(group_data)}")
            # Training data: all previous trials
            train_data = group_data.iloc[:i]
            X_train = np.stack(train_data['True_X_vec'].values, axis=0)  # shape (i, D)
            y_train = train_data['True_Y'].values.reshape(-1, 1)
            
            # Test point
            test_data = group_data.iloc[i]
            X_test = test_data['True_X_vec'].reshape(1, -1)
            llm_prediction = test_data['LLM_Prediction']
            
            if np.isnan(llm_prediction):
                # total_log_likelihood += (total_log_likelihood / i) # Use average log likelihood so far if LLM prediction is NaN
                continue # Skip if LLM prediction is NaN
            # Skip if we don't have enough training data
            if len(X_train) < 2:
                continue
            D = X_train.shape[1]   # number of input dimensions

            if kernel_fitter == 'SE':
                kernel = GPy.kern.RBF(input_dim=D, variance=1.0, lengthscale=fitter_length_scale)
            elif kernel_fitter == 'Matern½':
                kernel = GPy.kern.Exponential(input_dim=D, variance=1.0, lengthscale=fitter_length_scale)
            elif kernel_fitter == 'Matern1½':
                kernel = GPy.kern.Matern32(input_dim=D, variance=1.0, lengthscale=fitter_length_scale)
            elif kernel_fitter == 'Matern2½':
                kernel = GPy.kern.Matern52(input_dim=D, variance=1.0, lengthscale=fitter_length_scale)
            else:
                raise ValueError(f"Unknown kernel fitter: {kernel_fitter}")
            
            try:
                # Create and optimize GP model
                model = GPy.models.GPRegression(X_train, y_train, kernel)
                # Set reasonable bounds for hyperparameters
                model.kern.variance.fix(1.0)
                model.kern.lengthscale.fix(fitter_length_scale)
                model.Gaussian_noise.variance.fix(noise_val**2)
                y_pred, y_var = model.predict(X_test)
                y_std = np.sqrt(y_var)
                # Calculate probability density of LLM prediction under GP posterior
                if y_std[0, 0] > 1e-10:  # Avoid division by zero
                    log_prob = norm.logpdf(llm_prediction, loc=y_pred[0, 0], scale=y_std[0, 0])
                    total_log_likelihood += log_prob
                    n_predictions += 1
            except Exception as e:
                print(f"Warning: GP fitting failed for group {group_id}, trial {i}: {e}")
                continue
            if verbose:
                print("Log-likelihood for this prediction:", log_prob)
                print(f"Total log likelihood so far: {total_log_likelihood:.2f}, Predictions: {n_predictions}")
    return total_log_likelihood, n_predictions

def compare_kernels(df, fitter_length_scale: int = None, noise: bool = True):
    kernels_generator = {
        'SE': 'Radial Basis Function (Squared Exponential) λ=(1,8)',
        'Matern½': 'Matérn 1/2 (Exponential) λ=(1,8)',
    }
    kernels_fitter = {
        'Matern½': 'Matérn 1/2 (Exponential)',
        'Matern1½': 'Matérn 3/2',
        'Matern2½': 'Matérn 5/2',
        'SE': 'Radial Basis Function (Squared Exponential)',
    }
    results = {}
    for kernel_fitter, kernel_f_desc in kernels_fitter.items():
        for kernel_generator, kernel_g_desc in kernels_generator.items():
            if fitter_length_scale is None:
                fitter_length_scale = 1 if 'ls1' in kernel_generator else 8
            print(f"\nProcessing data generated from {kernel_g_desc} kernel with {kernel_f_desc} GP...")
            
            log_likelihood, n_preds = fit_gpy_and_calculate_likelihood(df, kernel_generator, kernel_fitter, fitter_length_scale, noise)
            
            results[f"G: {kernel_generator}\nF: {kernel_fitter}"] = {
                'log_likelihood': log_likelihood,
                'n_predictions': n_preds,
                'description': f"{kernel_g_desc} (Generator) {kernel_f_desc} (Fitter)"
            }
            print(f"{kernel_generator} (Generator) {kernel_fitter} (Fitter): Log-likelihood = {log_likelihood:.2f} ({n_preds} predictions)")
    return results

def compare_length_scales(df, fitter_length_scales: list, noise: bool = True):
    kernels_generator = {
        'SE': 'Radial Basis Function (Squared Exponential) λ=(1,8)',
        'Matern½': 'Matérn 1/2 (Exponential) λ=(1,8)',
    }
    results = {}
    for kernel_generator, kernel_g_desc in kernels_generator.items():
        results[kernel_g_desc] = {}
        for length_scale in fitter_length_scales:
            print(f"\nProcessing data generated from {kernel_g_desc} kernel fitted with Radial Basis Function (Squared Exponential) with length scale {length_scale}...")
        
            log_likelihood, n_preds = fit_gpy_and_calculate_likelihood(df, kernel_generator, 'SE', length_scale, noise)
        
            results[kernel_g_desc][length_scale] = {
                'log_likelihood': log_likelihood,
                'n_predictions': n_preds,
                'description': f"Length Scale: {length_scale} Generator: {kernel_g_desc}"
            }
            print(f"\nLength Scale {length_scale}: Log-likelihood = {log_likelihood:.2f} ({n_preds} predictions)")
    return results


def create_kernel_comparison_plots(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    kernel_names = list(results.keys())
    log_likelihoods = [results[k]['log_likelihood'] for k in kernel_names]
    colors = [
            '#0072B2',
            '#E69F00',
            '#56B4E9',
            '#F0E442',
            '#D55E00',
            '#009E73',
            '#CC79A7',
            '#000000',
        ]
    ax1.bar(kernel_names, log_likelihoods,
                   capsize=5, alpha=0.9, color=colors[:len(kernel_names)])
    ax1.set_ylabel('Sum of Log-Likelihoods')
    ax1.set_title('Total Likelihood of LLM Predictions')
    ax1.grid(True, alpha=0.3)
    n_predictions = [results[k]['n_predictions'] for k in kernel_names]
    avg_log_likelihoods = [ll/n if n > 0 else 0 for ll, n in zip(log_likelihoods, n_predictions)]
    
    ax2.bar(kernel_names, avg_log_likelihoods,
                    capsize=5, alpha=0.9, color=colors[:len(kernel_names)])
    
    ax2.set_ylabel('Average Log-Likelihood per Prediction')
    ax2.set_title('Average Likelihood Per LLM Prediction')
    ax2.grid(True, alpha=0.3)
    
    ax1.set_yscale('symlog', linthresh=400)
    ax2.set_yscale('symlog', linthresh=0.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_length_scale_comparison_plots(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    kernel_names = list(results.keys())
    length_scales = list(results[kernel_names[0]].keys())
    log_likelihoods_SE = [results['Radial Basis Function (Squared Exponential) λ=(1,8)'][ls]['log_likelihood'] for ls in length_scales]
    log_likelihoods_Matern = [results['Matérn 1/2 (Exponential) λ=(1,8)'][ls]['log_likelihood'] for ls in length_scales]
    
    colors = [
            '#0072B2',
            '#E69F00',
            '#56B4E9',
            '#F0E442',
            '#D55E00',
            '#009E73',
            '#CC79A7',
            '#000000',
        ]
    ax1.bar(length_scales, log_likelihoods_SE,
                   capsize=5, alpha=0.9, color=colors[:len(length_scales)])
    ax1.set_ylabel('Sum of Log-Likelihoods')
    ax1.set_xlabel('Fitter RBF Length Scale')
    ax1.set_title('Functions From Radial Basis Function (Squared Exponential λ=(1,8))\nKernel By Length Scale')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog', linthresh=400)
    
    ax2.bar(length_scales, log_likelihoods_Matern,
                   capsize=5, alpha=0.9, color=colors[:len(length_scales)])
    ax2.set_ylabel('Sum of Log-Likelihoods')
    ax2.set_xlabel('Fitter RBF Length Scale')
    ax2.set_title('Functions From Matern\nKernel (λ=(1,8)) By Length Scale')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('symlog', linthresh=400)
    
    plt.tight_layout()
    plt.show()
    return fig

@call_parse
def main(csv_file: str, 
         save_path_plots: str, 
         save_path_data: str
         ) -> None:
    
    BASE_PATH = "/llm-gaussian-processes/model_prediction_data/"
    model_type = csv_file.replace(BASE_PATH, '').split('/')[0] # Extract model type from path. BASE, GPs, GRPO, SFT.
    opt_reward_func = f"/{csv_file.replace(BASE_PATH, '').split('/')[1]}" if model_type == "GRPO" else ""
    # Load data
    df = load_and_prepare_data(csv_file)
    df = parse_multidim_inputs(df)
    
    print(f"Loaded {len(df)} data points")
    print(f"Unique functions: {len(df.groupby(['seed', 'kernel', 'noise', 'function_id']))}")
    
    # Compare kernels
    print("\n" + "="*60)
    print("KERNEL COMPARISON")
    print("="*60)
    
    results_kernels = compare_kernels(df)
    data_path = f"{save_path_data}{model_type}{opt_reward_func}/{csv_file.split('/')[-1].replace('.csv', '')}"
    with open(f"{data_path}_gp_kernel_comparison.json", 'w') as f:
        json.dump(results_kernels, f, indent=4)
    
    fig = create_kernel_comparison_plots(results_kernels)
    plot_path = f"{save_path_plots}{model_type}{opt_reward_func}/{csv_file.split('/')[-1].replace('.csv', '')}"
    plt.savefig(f"{plot_path}_gp_kernel_comparison.png", dpi=300)
    
    print("\n" + "="*60)
    print("KERNEL SUMMARY")
    print("="*60)
    
    best_kernel = max(results_kernels.keys(), key=lambda k: results_kernels[k]['log_likelihood'])
    
    for kernel_name in results_kernels.keys():
        r = results_kernels[kernel_name]
        is_best = "BEST" if kernel_name == best_kernel else ""
        print(f"{r['description']} ({kernel_name}):{is_best}")
        print(f"  Total Log-Likelihood: {r['log_likelihood']:.2f}")
        print(f"  Average per Prediction: {r['log_likelihood']/r['n_predictions']:.3f}")
        print(f"  Number of Predictions: {r['n_predictions']}")
        print()
    
    results_ls = compare_length_scales(df, [1,2,3,4,5,6,7,8])
    with open(f"{data_path}_gp_lengthscale_comparison.json", 'w') as f:
        json.dump(results_ls, f, indent=4)
    
    fig = create_length_scale_comparison_plots(results_ls)
    plt.savefig(f"{plot_path}_gp_lengthscale_comparison.png", dpi=300)

