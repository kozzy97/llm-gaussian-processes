import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

def gen_gp_dataset(seed: int,
                   length_scales: list[int] | int,
                   num_noise_values: int,
                   noise_level_mean: float,
                   noise_level_sd: float,
                   functions_per_noise: int = 100,
                   n_samples: int = 50,
                   xmin: float = 0.0,
                   xmax: float = 29.0,
                   n_grid: int = 900,
                   ):
    length_scales = [length_scales] if isinstance(length_scales, int) else length_scales
    np.random.seed(seed)
    random.seed(seed)
    X_grid = np.linspace(xmin, xmax, n_grid).reshape(-1, 1)
    noise_levels = []
    while len(noise_levels) < num_noise_values:
        noise = np.random.normal(noise_level_mean, noise_level_sd, size=1)
        if noise > 0:
            noise_levels.append(noise[0])
    dataset = []
    raw_funcs = { (ls,noise): [] for ls in length_scales for noise in noise_levels }
    raw_mae   = { (ls,noise): [] for ls in length_scales for noise in noise_levels }
    for ls in length_scales:
        # Matérn ν = 1/2 with same length scales as before
        prior_kernel = C(1.0, constant_value_bounds="fixed") * Matern(length_scale=ls, length_scale_bounds="fixed", nu=0.5)
        K_prior = prior_kernel(X_grid)
        for noise in noise_levels:
            for j in range(functions_per_noise):
                print(f"ls={ls}, noise={noise}, function {j+1}/{functions_per_noise}")
                K = K_prior + (noise**2) * np.eye(n_grid)
                K.flat[:: n_grid + 1] += 1e-10
                f = np.random.multivariate_normal(np.zeros(n_grid), K)
                f = MinMaxScaler((0,1)).fit_transform(f.reshape(-1,1)).ravel()
                raw_funcs[(ls,noise)].append(f)
                # Simulate active sampling with replacement & observational noise
                sampled_X = []
                sampled_Y = []
                mae_list  = []
                mae_old   = None
                mae_diffs = []
                for it in range(n_samples):
                    # sample idx w/ replacement
                    idx = random.randint(0, n_grid-1)
                    x_val = float(X_grid[idx,0])
                    y_noisy = f[idx] + np.random.normal(0, noise)
                    sampled_X.append([x_val])
                    sampled_Y.append(y_noisy)
                    # Keep the inference GP matched to the generator: Matérn ν=1/2
                    kern_fit = C(1.0, constant_value_bounds="fixed") * Matern(length_scale=ls, length_scale_bounds="fixed", nu=0.5)
                    gp = GaussianProcessRegressor(
                        kernel=kern_fit,
                        alpha=(noise**2),
                        optimizer=None,
                        normalize_y=False,
                        copy_X_train=True,
                        random_state=seed
                    )
                    gp.fit(np.array(sampled_X), np.array(sampled_Y))
                    mu = gp.predict(X_grid)  # mean only
                    mae = float(np.mean(np.abs(mu.ravel() - f)))
                    mae_list.append(mae)
                    if mae_old is not None:
                        mae_diffs.append(mae_old - mae)
                    mae_old = mae
                raw_mae[(ls,noise)].append(mae_list)
                # Build "text" from the actual sampled sequence
                lines = [f"X: {x[0]:.6f}, Y: {y:.6f}" for x,y in zip(sampled_X, sampled_Y)]
                line_str = ""
                for line, y in zip(lines, sampled_Y):
                    prompt = line_str + line.rsplit(" ", 1)[0] + " " # context so far plus "X: <x>, Y: "
                    line_str += f"{line}\n"
                    entry = {
                        "prompt":              prompt,
                        "ground_truth":        float(f"{y:.6f}"), # store as gold, same value as written in the text
                        "seed":                seed,
                        "kernel":              "Matern(nu=0.5)",
                        "length_scale":        ls,
                        "noise":               noise,
                        "n_samples":           n_samples,
                        "functions_per_noise": functions_per_noise,
                        "mae":                 mae_list,
                        "mae_diff":            mae_diffs
                    }
                    dataset.append(entry)
    return dataset
