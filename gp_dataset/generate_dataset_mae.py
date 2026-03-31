import numpy as np
import json
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, Matern
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
random.seed(seed)

xmin, xmax, n_grid = 0.0, 29.0, 900
X_grid = np.linspace(xmin, xmax, n_grid).reshape(-1, 1)

length_scale   = 8.0
noise_level    = 0.001
noise_var      = noise_level**2
functions_per_noise = 200
n_samples           = 100

kernels = [
    ("SE_ls1",      RBF(length_scale=1.0, length_scale_bounds="fixed")),
    ("SE_ls8",      RBF(length_scale=8.0, length_scale_bounds="fixed")),
    ("Matern_05_ls1", Matern(length_scale=1.0, nu=0.5, length_scale_bounds="fixed")),
    ("Matern_05_ls8", Matern(length_scale=8.0, nu=0.5, length_scale_bounds="fixed")),
]

dataset   = []
raw_funcs = { name: [] for name, _ in kernels }
raw_mae   = { name: [] for name, _ in kernels }

K_full = {}
for name, kern in kernels:
    K_full[name] = kern(X_grid, X_grid) + noise_var * np.eye(n_grid)

for name, kern in kernels:
    K = K_full[name]
    for fi in range(functions_per_noise):
        print(f"[{name}] sampling function {fi+1}/{functions_per_noise}")
        f = np.random.multivariate_normal(np.zeros(n_grid), K)
        f = MinMaxScaler((0,1)).fit_transform(f.reshape(-1,1)).ravel() # Change the scale
        raw_funcs[name].append(f)

        sampled_idxs = []
        sampled_Y   = []
        mae_list    = []
        mae_diffs   = []
        mae_old     = None

        for it in range(n_samples):
            idx = random.randint(0, n_grid-1)
            y0  = f[idx] + np.random.normal(0, noise_level)
            sampled_idxs.append(idx)
            sampled_Y.append(y0)

            idxs  = np.array(sampled_idxs)
            K_train = K[np.ix_(idxs, idxs)]
            K_train[np.diag_indices_from(K_train)] += noise_var
            K_star  = K[:, idxs]  # (900×n_obs)

            alpha = np.linalg.solve(K_train, np.array(sampled_Y))
            mu    = K_star.dot(alpha)    # (900,)

            mae = float(np.mean(np.abs(mu - f)))
            mae_list.append(mae)
            if mae_old is not None:
                mae_diffs.append(mae_old - mae)
            mae_old = mae

        raw_mae[name].append(mae_list)

        lines = [f"X: {X_grid[i,0]:.6f}, Y: {y:.6f}"
                 for i,y in zip(sampled_idxs, sampled_Y)]
        dataset.append({
            "text":                "\n".join(lines),
            "seed":                seed,
            "kernel":              name,
            "noise":               noise_level,
            "n_samples":           n_samples,
            "functions_per_noise": functions_per_noise,
            "mae":                 mae_list,
            "mae_diff":            mae_diffs
        })

with open("dataset_1d_mae_se_matern_ls_1_8.json", "w") as fp:
    json.dump(dataset, fp, indent=2)

for name, _ in kernels:
    picks = random.sample(range(functions_per_noise), 3)
    plt.figure()
    for p in picks:
        plt.plot(X_grid, raw_funcs[name][p])
    plt.title(f"Samples ({name})")
    plt.xlabel("X"); plt.ylabel("f(X)")
    plt.savefig(f"samples_{name}.png")
    plt.close()