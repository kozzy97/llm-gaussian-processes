import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastcore.script import call_parse
import GPy
from scipy.stats import norm, kstest
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    max_trials = df['Trial'].max()
    num_rows = len(df)
    df['function_id'] = [i for i in range((num_rows//max_trials)) for _ in range(1, max_trials+1)]
    return df

def _fit_tau2(residuals, variances):
    residuals = np.asarray(residuals, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if len(residuals) == 0:
        return 0.0

    def nll(log_tau2):
        tau2 = np.exp(log_tau2)
        s2 = variances + tau2
        return 0.5*np.sum(np.log(s2) + (residuals**2)/s2)

    res = minimize_scalar(nll, bounds=(-25, 10), method='bounded')  # ~[1e-11, ~22k]
    return float(np.exp(res.x))

def fit_gpy_and_calculate_likelihood(df, kernel_generator, kernel_fitter, fitter_length_scale: int, noise: bool, verbose: bool = False):
    k_generators = ['SE_ls1', 'SE_ls8'] if kernel_generator.startswith('SE') else ['Matern_05_ls1', 'Matern_05_ls8']
    df = df[df['kernel'].isin(k_generators)]  # Filter by kernel type used to generate the data
    
    noise_val = df['noise'].unique()[0] if not df['noise'].isnull().all() and noise else 0.0
    function_groups = df.groupby(['seed', 'kernel', 'noise', 'function_id'])
    residuals, variances, pit_vals = [], [], []
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
            X_train = train_data['True_X'].values.reshape(-1, 1)
            y_train = train_data['True_Y'].values.reshape(-1, 1)
            # Test point
            test_data = group_data.iloc[i]
            X_test = np.array([[test_data['True_X']]])
            llm_prediction = test_data['LLM_Prediction']
            if np.isnan(llm_prediction):
                continue
            # Skip if we don't have enough training data
            if len(X_train) < 2:
                continue
            # Set up GP kernel
            if kernel_fitter == 'SE': # Filter by kernel type used to fit the GP
                kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=fitter_length_scale)
            elif kernel_fitter == 'Matern½':
                kernel = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=fitter_length_scale)
            elif kernel_fitter == 'Matern1½':
                kernel = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=fitter_length_scale)
            elif kernel_fitter == 'Matern2½':
                kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=fitter_length_scale)
            else:
                raise ValueError(f"Unknown kernel fitter: {kernel_fitter}")
            try:
                # Create and optimize GP model
                model = GPy.models.GPRegression(X_train, y_train, kernel)
                # Set reasonable bounds for hyperparameters
                model.kern.variance.fix(1.0)
                model.kern.lengthscale.fix(fitter_length_scale)
                model.Gaussian_noise.variance.fix(noise_val**2)
                y_pred, y_var = model.predict(X_test, include_likelihood=True)
                mu = float(y_pred[0, 0])
                s2 = float(y_var[0, 0])
                if s2 <= 1e-12:
                    continue
                r = float(llm_prediction - mu)
                z = r / np.sqrt(s2)
                residuals.append(r)
                variances.append(s2)
                pit_vals.append(norm.cdf(z))
                n_predictions += 1

            except Exception as e:
                print(f"Warning: GP fitting failed for group {group_id}, trial {i}: {e}")
                continue
    return np.array(residuals), np.array(variances), np.array(pit_vals), n_predictions

def tau_adjusted_loglik(residuals, variances, tau2):
    if len(residuals) == 0:
        return 0.0
    s2 = variances + tau2
    return float(np.sum(-0.5*(np.log(2*np.pi*s2) + (residuals**2)/s2)))

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
    ls_default = 1 if fitter_length_scale is None else int(fitter_length_scale)

    all_resid, all_vars = [], []
    for kernel_fitter in kernels_fitter.keys():
        for kernel_generator in kernels_generator.keys():
            rs, vs, _, n = fit_gpy_and_calculate_likelihood(
                df, kernel_generator, kernel_fitter, ls_default, noise, verbose=False
            )
            all_resid.append(rs)
            all_vars.append(vs)
    pooled_resid = np.concatenate([r for r in all_resid if len(r) > 0]) if len(all_resid) else np.array([])
    pooled_vars = np.concatenate([v for v in all_vars if len(v) > 0]) if len(all_vars) else np.array([])
    tau2 = _fit_tau2(pooled_resid, pooled_vars)
    print(f"\nEstimated shared SAM noise tau^2 = {tau2:.6g}")

    for kernel_fitter, kernel_f_desc in kernels_fitter.items():
        for kernel_generator, kernel_g_desc in kernels_generator.items():
            print(f"\nProcessing data generated from {kernel_g_desc} kernel with {kernel_f_desc} GP...")
            
            resid, vars_, pit, n_preds = fit_gpy_and_calculate_likelihood(
                df, kernel_generator, kernel_fitter, ls_default, noise, verbose=False
            )
            ll_tau = tau_adjusted_loglik(resid, vars_, tau2)
            pit_stat = float(kstest(pit, 'uniform').statistic) if len(pit) > 0 else None
            
            results[f"G: {kernel_generator}\nF: {kernel_fitter}"] = {
                'log_likelihood': ll_tau,           # tau^2-adjusted
                'n_predictions': int(n_preds),
                'description': f"{kernel_g_desc} (Generator) {kernel_f_desc} (Fitter)",
                'tau2': float(tau2),
                'pit_ks': pit_stat
            }
            print(f"{kernel_generator} (Generator) {kernel_fitter} (Fitter): "
                  f"Log-likelihood(τ²) = {ll_tau:.2f} ({n_preds} predictions), "
                  f"PIT KS = {pit_stat if pit_stat is not None else 'NA'}")  
    return results

def compare_length_scales(df, fitter_length_scales: list, noise: bool = True):
    """
    Compare different length scales for a fixed kernel
    """
    kernels_generator = {
        'SE': 'Radial Basis Function (Squared Exponential) λ=(1,8)',
        'Matern½': 'Matérn 1/2 (Exponential) λ=(1,8)',
    }
    results = {}
    all_resid, all_vars = [], []
    for kernel_generator in kernels_generator.keys():
        for length_scale in fitter_length_scales:
            rs, vs, _, _ = fit_gpy_and_calculate_likelihood(
                df, kernel_generator, 'SE', int(length_scale), noise, verbose=False
            )
            all_resid.append(rs)
            all_vars.append(vs)

    pooled_resid = np.concatenate([r for r in all_resid if len(r) > 0]) if len(all_resid) else np.array([])
    pooled_vars = np.concatenate([v for v in all_vars if len(v) > 0]) if len(all_vars) else np.array([])
    tau2 = _fit_tau2(pooled_resid, pooled_vars)
    print(f"\n[Length-scale comparison] Estimated shared SAM noise tau^2 = {tau2:.6g}")

    for kernel_generator, kernel_g_desc in kernels_generator.items():
        results[kernel_g_desc] = {}
        for length_scale in fitter_length_scales:
            print(f"\nProcessing data generated from {kernel_g_desc} kernel fitted with Radial Basis Function (Squared Exponential) with length scale {length_scale}...")
        
            resid, vars_, pit, n_preds = fit_gpy_and_calculate_likelihood(
                df, kernel_generator, 'SE', int(length_scale), noise, verbose=False
            )
            ll_tau = tau_adjusted_loglik(resid, vars_, tau2)
            results[kernel_g_desc][int(length_scale)] = {
                'log_likelihood': ll_tau,
                'n_predictions': int(n_preds),
                'description': f"Length Scale: {length_scale} Generator: {kernel_g_desc}",
                'tau2': float(tau2),
                'pit_ks': float(kstest(pit, 'uniform').statistic) if len(pit) > 0 else None
            }
            print(f"Length Scale {length_scale}: Log-likelihood(τ²) = {ll_tau:.2f} ({n_preds} predictions)")
    return results


def create_kernel_comparison_plots(results):
    """
    Create comprehensive comparison plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    # Get kernel names and log-likelihoods
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
    ax1.set_ylabel('Sum of Log-Likelihoods (τ²-adjusted)')
    ax1.set_title('Total Likelihood of LLM Predictions (τ²-adjusted)')
    ax1.grid(True, alpha=0.3)
    n_predictions = [results[k]['n_predictions'] for k in kernel_names]
    avg_log_likelihoods = [ll/n if n > 0 else 0 for ll, n in zip(log_likelihoods, n_predictions)]
    
    ax2.bar(kernel_names, avg_log_likelihoods,
                    capsize=5, alpha=0.9, color=colors[:len(kernel_names)])
    
    ax2.set_ylabel('Average Log-Likelihood per Prediction (τ²-adjusted)')
    ax2.set_title('Average Likelihood Per LLM Prediction (τ²-adjusted)')
    ax2.grid(True, alpha=0.3)
    
    ax1.set_yscale('symlog', linthresh=400)
    ax2.set_yscale('symlog', linthresh=0.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_length_scale_comparison_plots(results):
    """
    Create length scale comparison plots
    """
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
    ax1.set_ylabel('Sum of Log-Likelihoods (τ²-adjusted)')
    ax1.set_xlabel('Fitter RBF Length Scale')
    ax1.set_title('Functions From Radial Basis Function (Squared Exponential λ=(1,8))\nKernel By Length Scale')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog', linthresh=400)
    
    ax2.bar(length_scales, log_likelihoods_Matern,
                   capsize=5, alpha=0.9, color=colors[:len(length_scales)])
    ax2.set_ylabel('Sum of Log-Likelihoods (τ²-adjusted)')
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
    """
    Main analysis function
    """
    BASE_PATH = "/llm-gaussian-processes/model_prediction_data/"
    model_type = csv_file.replace(BASE_PATH, '').split('/')[0] # Extract model type from path. BASE, GPs, GRPO, SFT.
    opt_reward_func = f"/{csv_file.replace(BASE_PATH, '').split('/')[1]}" if model_type == "GRPO" else ""
    # Load data
    df = load_and_prepare_data(csv_file)
    
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
    
    # Create plots
    fig = create_kernel_comparison_plots(results_kernels)
    plot_path = f"{save_path_plots}{model_type}{opt_reward_func}/{csv_file.split('/')[-1].replace('.csv', '')}"
    plt.savefig(f"{plot_path}_gp_kernel_comparison.png", dpi=300)
    
    # Print summary
    print("\n" + "="*60)
    print("KERNEL SUMMARY")
    print("="*60)
    
    best_kernel = max(results_kernels.keys(), key=lambda k: results_kernels[k]['log_likelihood'])
    
    for kernel_name in results_kernels.keys():
        r = results_kernels[kernel_name]
        is_best = " BEST" if kernel_name == best_kernel else ""
        print(f"{r['description']} ({kernel_name}):{is_best}")
        print(f"  Total Log-Likelihood: {r['log_likelihood']:.2f}")
        print(f"  Average per Prediction: {r['log_likelihood']/r['n_predictions']:.3f}")
        print(f"  Number of Predictions: {r['n_predictions']}")
        print()
    
    results_ls = compare_length_scales(df, [1,2,3,4,5,6,7,8])
    with open(f"{data_path}_gp_lengthscale_comparison.json", 'w') as f:
        json.dump(results_ls, f, indent=4)
    
    # Create plots
    fig = create_length_scale_comparison_plots(results_ls)
    plt.savefig(f"{plot_path}_gp_lengthscale_comparison.png", dpi=300)

