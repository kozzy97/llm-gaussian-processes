import pandas as pd, numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("wilks_results.csv")
df["log10_BF"] = (df["W"] / 2) / np.log(10)

heat = (
    df.groupby(["model", "kernel"])["log10_BF"]
      .mean()
      .unstack()
)

# nicer row ordering if model names contain checkpoint steps
heat = heat.reindex(
    sorted(heat.index,
           key=lambda s: int(s.split("checkpoint-")[-1].split("_")[0])
                         if "checkpoint-" in s else -1)
)

plt.figure(figsize=(12, 8))
im = plt.imshow(
    heat.clip(-10, 10), # clip extremes
    cmap="RdBu_r",
    vmin=-10, vmax=10,
    aspect="auto"
)

plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(heat.index)),  heat.index,  fontsize=8)

cbar = plt.colorbar(im)
cbar.set_label("log$_{10}$ Bayes factor  (model / GP oracle)")

plt.title("Evidence comparison: red = LLM favoured | blue = GP favoured")
plt.tight_layout()
plt.savefig('wilks_heatmap_bf.png')
