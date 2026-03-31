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
    """
    Parses the dataset entry for n-dimensional inputs.
    It assumes the text format is: "X0: val, X1: val, ..., Y: val"
    """
    text_data = data_entry["text"]
    lines = [ln.strip() for ln in text_data.split("\n") if ln.strip()]
    
    prompts, x_strs, y_vals, ids = [], [], [], []
    example = 0
    
    # Updated prompt to reflect multiple inputs
    accumulated = (
        "You are a function approximator. I will give you a set of input variables (X), "
        "and you need to predict the output value (Y). "
        "There may be noise in the true prediction. Your task is to provide your best estimate for Y. "
        "Provide that and only that, without any additional text.\n"
    )
    
    for line in lines:
        # We split by the last instance of "Y:" to separate inputs from output
        # This handles both "X: 1.0, Y: 2.0" and "X0: 1.0, X1: 2.0, Y: 3.0"
        if "Y:" not in line:
            continue
            
        # Split into [Inputs part, Output part]
        # e.g. "X0: 0.1, X1: 0.2" and "0.55"
        parts = line.split("Y:")
        
        # Clean up the X string (remove trailing commas/spaces)
        x_part = parts[0].strip().rstrip(",")
        y_part = parts[1].strip()
        
        try:
            y_val = float(y_part)
        except ValueError:
            continue
        
        example += 1
        
        # The prompt asks the model for Y given the current history + current X
        prompt = accumulated + f"{x_part}, Y: "
        
        prompts.append(prompt)
        x_strs.append(x_part) # Store X as a string representation of the vector
        y_vals.append(y_val)
        ids.append(example)
        
        # Update history for the next iteration
        accumulated += f"{x_part}, Y: {y_val}\n"
        
    return prompts, x_strs, y_vals, ids

def main(model_name: str, data_file: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    max_seq_length = 8192
    
    # 1) Load Model with dynamic 4-bit quantization
    print(f"Loading model: {model_name}")

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
    # Sanitize model name for filename
    model_base = model_name.rstrip("/").split("/")[-1]
    data_base = os.path.splitext(os.path.basename(data_file))[0]
    
    out_csv = os.path.join(save_dir, f"{model_base}_{data_base}_results.csv")
    
    # Note: True_X is now a string to accommodate scalar or vectors
    header = ['Trial','True_X_Str','True_Y','LLM_Prediction','Absolute_Error','seed','kernel','noise', 'dimension']
    
    write_header = not os.path.exists(out_csv)
    csv_file = open(out_csv, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)
        csv_file.flush()
    
    print(f"Running ICL evaluation, saving to {out_csv}…")
    
    for _, entry in enumerate(data):
        # Process the ICL sequence
        prompts, xs, ys, ids = process_icl_data(entry)
        
        # Limit to first 50 samples per function if desired
        limit = 50
        prompts, xs, ys, ids = prompts[:limit], xs[:limit], ys[:limit], ids[:limit]
        
        for trial, prompt, true_x_str, true_y in zip(ids, prompts, xs, ys):
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32, # Short generation for just the number
                temperature = 1.0, 
                top_p = 1.0,
            )
            
            # Slicing off the prompt length to get just the response
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            
            # Handle Thinking tokens (DeepSeek/Qwen specific <think> tags)
            # 151668 is often the end_of_thought token in Qwen 2.5 math/logic finetunes
            try:
                # Try to find the closing think tag
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            
            # Robust regex to find the first floating point number in the response
            m = re.search(r"[-+]?(?:\d*\.\d+|\d+)", content)
            if m:
                try:
                    pred_val = float(m.group(0))
                    err = abs(true_y - pred_val)
                except ValueError:
                     pred_val, err = 'ParseErr', 'N/A'
            else:
                pred_val, err = 'N/A', 'N/A'
            
            writer.writerow([
                trial, 
                true_x_str, 
                true_y,
                pred_val, 
                err,
                entry.get('seed'), 
                entry.get('kernel'),
                entry.get('noise'),
                entry.get('dimension', '1') # Default to 1 if missing
            ])
            csv_file.flush()
    
    csv_file.close()
    print(f"Completed. Results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_hf_path", type=str,
        default="unsloth/Qwen3-8B-unsloth-bnb-4bit", # Example default
        help="HuggingFace model path or Unsloth path"
    )
    parser.add_argument(
        "--data_file_path", type=str,
        required=True,
        help="Path to the JSON dataset file generated by the previous script"
    )
    parser.add_argument(
        "--save_path", type=str,
        default="./results",
        help="Directory to save CSV results"
    )
    args = parser.parse_args()
    
    main(
        args.model_hf_path,
        args.data_file_path,
        args.save_path
    )