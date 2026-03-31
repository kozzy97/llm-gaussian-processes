import os
import re
import json
import csv
import torch
import argparse

from unsloth import FastLanguageModel
# from transformers import logging
# from transformers import TextStreamer

# silence unnecessary warnings
# logging.set_verbosity_error()

def process_icl_data(data_entry):
    text_data = data_entry["text"]
    lines = [ln.strip() for ln in text_data.split("\n") if ln.strip()]
    
    prompts, x_vals, y_vals, ids = [], [], [], []
    example = 0
    accumulated = (
        "You are a number predictor. I will give you a number, X, and then you need to predict a new number, Y. "
        "There may be noise in the true prediction. Your task is to provide your best estimate for Y. Provide that and only that, without any additional text.\n"
    )
    for line in lines:
        m = re.match(r"X: ([\d\.]+), Y: (-?[\d\.]+)", line)
        if not m:
            continue
        x_val, y_val = m.groups()
        example += 1
        prompt = accumulated + f"X: {x_val}, Y: "
        prompts.append(prompt)
        x_vals.append(float(x_val))
        y_vals.append(float(y_val))
        ids.append(example)
        accumulated += f"X: {x_val}, Y: {y_val}\n"
    return prompts, x_vals, y_vals, ids

def main(model_name: str, data_file: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    max_seq_length = 8192
    
    # 1) Load Qwen3 with dynamic 4-bit quantization
    print(model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        load_in_8bit=False,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)
    
    # 2) Prepare CSV writer
    model_name = model_name.replace("/checkpoint-", "-checkpoint-")
    base = os.path.basename(model_name).replace("/", "_")
    out_csv = os.path.join(save_dir, f"{base}.csv")
    header = ['Trial','True_X','True_Y','LLM_Prediction','Absolute_Error','seed','kernel','noise']
    write_header = not os.path.exists(out_csv)
    csv_file = open(out_csv, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)
        csv_file.flush()
    
    print(f"Running ICL with {model_name}…")
    
    for _, entry in enumerate(data):
        prompts, xs, ys, ids = process_icl_data(entry)
        prompts, xs, ys, ids = prompts[:50], xs[:50], ys[:50], ids[:50]
        
        for trial, prompt, true_x, true_y in zip(ids, prompts, xs, ys):
            text = tokenizer.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10,
                temperature = 1.0, top_p = 1.0,
                # streamer = text_streamer,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            m = re.search(r"[-+]?(?:\d*\.\d+|\d+)", content)
            if m:
                pred_val = float(m.group(0))
                err = abs(true_y - pred_val)
            else:
                pred_val, err = 'N/A', 'N/A'
            
            writer.writerow([
                trial, true_x, true_y,
                pred_val, err,
                entry.get('seed'), entry.get('kernel'),
                entry.get('noise')
            ])
            csv_file.flush()
    
    csv_file.close()
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_hf_path", type=str,
        default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    )
    parser.add_argument(
        "--data_file_path", type=str,
        default="/gp-icl/data/dataset_1d_mse_se_matern_ls_1_8.json"
    )
    parser.add_argument(
        "--save_path", type=str,
        default="/gp-icl/unsloth/data_qwen3",
    )
    args = parser.parse_args()
    main(
        args.model_hf_path,
        args.data_file_path,
        args.save_path
    )