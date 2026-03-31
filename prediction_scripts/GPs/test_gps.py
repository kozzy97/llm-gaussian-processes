import argparse
import csv
import json
import os
import re

import numpy as np
import GPy

def process_icl_data(data_entry):
    text_data = data_entry["text"]
    lines = [ln.strip() for ln in text_data.split("\n") if ln.strip()]
    
    x_vals, y_vals, ids = [], [], []
    example = 0
    for line in lines:
        m = re.match(r"X: ([\d\.]+), Y: (-?[\d\.]+)", line)
        if not m:
            continue
        x_val, y_val = m.groups()
        example += 1
        x_vals.append(float(x_val))
        y_vals.append(float(y_val))
        ids.append(example)
    return x_vals, y_vals, ids

def gp_predict(kernel,
               out_csv: str,
               data_file: str,
               ):
    header = ['Trial','True_X','True_Y','LLM_Prediction','Absolute_Error','seed','kernel','noise', 'LLM_Prediction_Var']
    write_header = not os.path.exists(out_csv)
    csv_file = open(out_csv, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)
        csv_file.flush()

    with open(data_file, 'r') as f:
        data = json.load(f)

    for entry in data:
        xs, ys, ids = process_icl_data(entry)
        xs, ys, ids = xs[:50], ys[:50], ids[:50]   # only first 50 pairs

        results = []
        for i, (x_test, y_true) in enumerate(zip(xs, ys)):
            x_test_arr = np.array([[x_test]])
            
            if i == 0:
                mean_pred = 0.0
                var_pred  = kernel.K(x_test_arr).item()
            else:
                X_train = np.array(xs[:i])[:, None]   # shape (i,1)
                Y_train = np.array(ys[:i])[:, None]

                k = kernel.copy()
                m = GPy.models.GPRegression(X_train, Y_train, k)
                
                m.optimize(messages=False)
                
                mean_pred, var_pred = m.predict(x_test_arr)
                mean_pred = mean_pred.item()
                var_pred  = var_pred.item()
            
            err = abs(y_true - mean_pred)

            writer.writerow([
                ids[i], x_test, y_true,
                mean_pred, err,
                entry.get('seed'), entry.get('kernel'),
                entry.get('noise'), var_pred
            ])
            csv_file.flush()
    csv_file.close()
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", type=str,
        default="matern12",
    )
    parser.add_argument(
        "--lengthscale", type=int,
        default=1,
    )
    parser.add_argument(
        "--save_path", type=str,
        default="/llm-gaussian-processes/model_prediction_data/GPs",
    )
    parser.add_argument(
        "--data_file_path", type=str,
        default="/llm-gaussian-processes/gp_dataset/dataset_1d_mae_se_matern_ls_1_8.json"
    )
    args = parser.parse_args()
    if args.kernel == "matern12":
        kernel = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=args.lengthscale)
    elif args.kernel == "matern32":
        kernel = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=args.lengthscale)
    elif args.kernel == "matern52":
        kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=args.lengthscale)
    elif args.kernel == "SE" or args.kernel == "RBF":
        kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=args.lengthscale)
    else:
        raise ValueError(f"Unknown kernel: {args.kernel}. Only 'matern12', 'matern32', 'matern52', and 'SE'/'RBF' are supported.")
    gp_predict(
        kernel,
        os.path.join(
            args.save_path, f"gp_predictions_{args.kernel}_ls_{args.lengthscale}.csv"
        ),
        args.data_file_path,
    )
