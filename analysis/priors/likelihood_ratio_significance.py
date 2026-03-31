import json, glob, argparse, math, pathlib
import pandas as pd
from scipy.stats import chi2

def load_jsons(dir_path: str, pattern: str):
    base = pathlib.Path(dir_path)
    files = list(base.rglob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched '{pattern}' under '{dir_path}'.")
    print(f"[info] found {len(files)} matching JSONs")
    data = {}
    for fp in files:
        name = fp.stem.replace("_gp_kernel_comparison", "")
        with open(fp) as f:
            obj = json.load(f)
        obj["__file_path__"] = str(fp)
        data[name] = obj
    return data

def wilks(gp_ll, model_ll):
    W = -2 * (gp_ll - model_ll)
    p = 1.0 - chi2.cdf(W, df=1)
    return W, p

def main(json_dir, pattern, save_csv):
    jdata = load_jsons(json_dir, pattern)

    kernel_keys = next(iter(jdata.values())).keys() - {"__file_path__"}

    gp_files    = {k:v for k,v in jdata.items() if k.startswith("gp_predictions")}
    model_files = {k:v for k,v in jdata.items() if not k.startswith("gp_predictions")}

    if not gp_files:
        raise RuntimeError("No GP oracle JSONs (filenames starting "
                           "'gp_predictions') found.")

    rows = []
    for model_name, mres in model_files.items():
        for gp_name, gres in gp_files.items():
            for kern in kernel_keys:
                ll_model = mres[kern]["log_likelihood"]
                ll_gp    = gres[kern]["log_likelihood"]
                W, p = wilks(ll_gp, ll_model)
                rows.append({
                    "model": model_name,
                    "gp_null": gp_name,
                    "kernel": kern,
                    "W": W,
                    "p_value": p
                })

    df = pd.DataFrame(rows).sort_values(["model", "kernel"])
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\n[saved] Wilks results → {save_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True,
                        help="root folder that contains the *.json files")
    parser.add_argument("--pattern", default="*_gp_kernel_comparison.json",
                        help="glob pattern, e.g. '**/*_gp_kernel_comparison.json'")
    parser.add_argument("--save_csv", default=None,
                        help="optional CSV output")
    args = parser.parse_args()
    main(**vars(args))

