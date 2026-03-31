import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv("wilks_results.csv")

heat = df.groupby(["model", "kernel"])["W"].mean().unstack()
heat_log = np.sign(heat) * np.log10(np.abs(heat) + 1)

# clip extremes
heat_clip = heat_log.clip(-10, 10)

# diverging heat‑map
plt.figure(figsize=(12, 8))
im = plt.imshow(heat_clip, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")

plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(heat.index)), heat.index, fontsize=8)
plt.colorbar(im, label="signed log₁₀|W|  (Δ fit vs GP oracle)")
plt.title("Wilks statistic (log-compressed; red=model>GP, blue=GP>model)")
plt.tight_layout()
plt.savefig('wilks_heatmap.png')