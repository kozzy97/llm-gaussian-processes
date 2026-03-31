import numpy as np
import json
import random
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, Matern

seed = 42
np.random.seed(seed)
random.seed(seed)

# Configuration
target_dims = [3, 4]  # The dimensions to generate datasets for. Should work for 1D as well.
xmin, xmax = 0.0, 29.0
target_total_points = 2500  # Keep total grid size manageable to prevent RAM crash

# Hyper-params
noise_level    = 0.001
noise_var      = noise_level**2
functions_per_noise = 200  
n_samples           = 100

kernels_config = [
    ("SE_ls1",        RBF(length_scale=1.0, length_scale_bounds="fixed")),
    ("SE_ls8",        RBF(length_scale=8.0, length_scale_bounds="fixed")),
    ("Matern_05_ls1", Matern(length_scale=1.0, nu=0.5, length_scale_bounds="fixed")),
    ("Matern_05_ls8", Matern(length_scale=8.0, nu=0.5, length_scale_bounds="fixed")),
]

def generate_grid(n_dim, x_min, x_max, total_points):
    # Calculate points per dimension: N^(1/d)
    pts_per_dim = int(total_points ** (1 / n_dim))
    # Ensure at least a minimal grid
    pts_per_dim = max(pts_per_dim, 4) 
    
    print(f"--- Generatig {n_dim}D Grid: {pts_per_dim} points per dim "
          f"(Total: {pts_per_dim**n_dim}) ---")
    
    # Create linspace for each dimension
    linspaces = [np.linspace(x_min, x_max, pts_per_dim) for _ in range(n_dim)]
    
    # Create meshgrid (indexing='ij' matrix indexing)
    mesh = np.meshgrid(*linspaces, indexing='ij')
    
    # Flatten and stack: result shape (total_points, n_dim)
    X_grid = np.stack([m.ravel() for m in mesh], axis=-1)
    
    # Return grid and the shape tuple for plotting reconstruction later
    grid_shape = tuple([pts_per_dim] * n_dim)
    return X_grid, grid_shape

# --- Main Loop over Dimensions ---
for n_dim in target_dims:
    print(f"\nProcessing Dimension: {n_dim}")
    
    X_grid, grid_shape = generate_grid(n_dim, xmin, xmax, target_total_points)
    n_grid = X_grid.shape[0]
    
    dataset = []
    raw_funcs = {name: [] for name, _ in kernels_config}
    raw_mae = {name: [] for name, _ in kernels_config}
    
    # Precompute full covariance matrices for this specific grid/dimension
    K_full = {}
    for name, kern in kernels_config:
        # kern(X, X) handles (N, D) inputs automatically
        K_full[name] = kern(X_grid, X_grid) + noise_var * np.eye(n_grid)
    
    for name, kern in kernels_config:
        K = K_full[name]
        
        for fi in range(functions_per_noise):
            if fi % 10 == 0:
                print(f"  [{name}] {n_dim}D function {fi+1}/{functions_per_noise}")
            
            # 1. Draw latent f ~ N(0, K)
            f = np.random.multivariate_normal(np.zeros(n_grid), K)
            
            # 2. Scale to [0,1]
            f = MinMaxScaler((0, 1)).fit_transform(f.reshape(-1, 1)).ravel()
            raw_funcs[name].append(f)
            
            sampled_idxs = []
            sampled_Y = []
            mae_list = []
            mae_diffs = []
            mae_old = None
            
            unique_indices = np.random.choice(n_grid, size=n_samples, replace=False)
            
            # 3. Active Sampling Loop
            for it in range(n_samples):
                # Use the pre-selected unique index instead of random.randint
                idx = unique_indices[it] 
                
                # The rest of your logic remains exactly the same
                y0 = f[idx] + np.random.normal(0, noise_level)
                sampled_idxs.append(idx)
                sampled_Y.append(y0)
                
                # GP Posterior Update
                idxs = np.array(sampled_idxs)
                
                # K_train: Covariance of sampled points
                K_train = K[np.ix_(idxs, idxs)]
                K_train[np.diag_indices_from(K_train)] += noise_var
                
                # K_star: Covariance between all grid points and sampled points
                K_star = K[:, idxs]  # (n_grid, n_obs)
                
                # Solve for mean
                # (We assume zero mean prior for simplicity)
                try:
                    alpha = np.linalg.solve(K_train, np.array(sampled_Y))
                    mu = K_star.dot(alpha)
                except np.linalg.LinAlgError:
                    # Fallback if singular (rare with jitter)
                    mu = np.zeros(n_grid)
                
                # Calculate MAE over the whole field
                mae = float(np.mean(np.abs(mu - f)))
                mae_list.append(mae)
                if mae_old is not None:
                    mae_diffs.append(mae_old - mae)
                mae_old = mae
            
            raw_mae[name].append(mae_list)
            
            # 4. Text formatting for N-dimensions
            # Format: "X0: 1.23, X1: 4.56, ... Y: 0.78"
            lines = []
            for i, y in zip(sampled_idxs, sampled_Y):
                coords_str = ", ".join(
                    [f"X{d}: {X_grid[i, d]:.4f}" for d in range(n_dim)]
                )
                lines.append(f"{coords_str}, Y: {y:.4f}")
            
            dataset.append({
                "text": "\n".join(lines),
                "seed": seed,
                "dimension": n_dim,
                "kernel": name,
                "noise": noise_level,
                "n_samples": n_samples,
                "functions_per_noise": functions_per_noise,
                "mae": mae_list,
                "mae_diff": mae_diffs
            })
    
    # Save JSON per dimension
    out_file = f"dataset_{n_dim}d_mae_se_matern_ls_1_8.json"
    with open(out_file, "w") as fp:
        json.dump(dataset, fp, indent=2)