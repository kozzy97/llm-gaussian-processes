import argparse
import gc
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import pandas as pd

from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from generate_dataset import gen_gp_dataset

def process_dataset(dataset: Dataset) -> Dataset:
    problems = dataset["prompt"]
    solutions = dataset["ground_truth"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",       "content" : problem},
            {"role" : "assistant",  "content" : f"{solution}"},
        ])
    return {"conversations" : conversations}

def main(seed: int,
        length_scales: list[int] | int,
        num_noise_values: int,
        noise_level_mean: float,
        noise_level_sd: float,
        model_name: str,
        output_dir: str,
        functions_per_noise: int = 100,
        n_samples: int = 50,
        xmin: float = 0.0,
        xmax: float = 29.0,
        n_grid: int = 900) -> None:
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 3000,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = 32,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )
    
    dataset = Dataset.from_list(gen_gp_dataset(seed, length_scales, num_noise_values, noise_level_mean, noise_level_sd, functions_per_noise=functions_per_noise, n_samples=n_samples, xmin=xmin, xmax=xmax, n_grid=n_grid))
    dataset = tokenizer.apply_chat_template(
                                dataset.map(process_dataset, batched = True)["conversations"],
                                tokenize = False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
    dataset = pd.concat([pd.Series(dataset)])
    dataset.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    dataset = dataset.shuffle(seed=seed)
    
    model = FastLanguageModel.get_peft_model(
                                model,
                                r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
                                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                "gate_proj", "up_proj", "down_proj",],
                                lora_alpha = 32,  # Best to choose alpha = rank or rank*2
                                lora_dropout = 0, # Supports any, but = 0 is optimized
                                bias = "none",    # Supports any, but = "none" is optimized
                                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                                random_state = seed,
                                use_rslora = False,   # We support rank stabilized LoRA
                                loftq_config = None,  # And LoftQ
                            )
    
    trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                max_steps = 10000,
                save_steps = 500,
                learning_rate = 2e-5,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = seed,
                report_to = "none", # Use this for WandB etc
                output_dir = output_dir,
            ),
        )
    
    trainer.train()

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()    
    
    argparser = argparse.ArgumentParser(description="Train QLoRA SFT on a GP dataset.")
    argparser.add_argument("--seed",                type=int,   required=True,                help="Random seed for dataset generation.")
    argparser.add_argument("--length_scales",       type=int,   nargs='+',    required=True,  help="Length scales for the GP kernel.")
    argparser.add_argument("--num_noise_values",    type=int,   required=True,                help="Number of noise values to sample.")
    argparser.add_argument("--noise_level_mean",    type=float, required=True,                help="Mean of the noise level.")
    argparser.add_argument("--noise_level_sd",      type=float, required=True,                help="Standard deviation of the noise level.")
    argparser.add_argument("--model_name",          type=str,   required=True,                help="Model name for QLoRA.")
    argparser.add_argument("--output_dir",          type=str,   required=True,                help="Output directory for the model.")
    argparser.add_argument("--functions_per_noise", type=int,   default=100,                  help="Number of functions per noise level.")
    argparser.add_argument("--n_samples",           type=int,   default=50,                   help="Number of samples to draw from the GP.")
    argparser.add_argument("--xmin",                type=float, default=0.0,                  help="Minimum x value for the GP.")
    argparser.add_argument("--xmax",                type=float, default=29.0,                 help="Maximum x value for the GP.")
    argparser.add_argument("--n_grid",              type=int,   default=900,                  help="Number of grid points for the GP.")
    args = argparser.parse_args()
    
    main(
        seed=                   args.seed,
        length_scales=          args.length_scales,
        num_noise_values=       args.num_noise_values,
        noise_level_mean=       args.noise_level_mean,
        noise_level_sd=         args.noise_level_sd,
        model_name=             args.model_name,
        output_dir=             args.output_dir,
        functions_per_noise=    args.functions_per_noise,
        n_samples=              args.n_samples,
        xmin=                   args.xmin,
        xmax=                   args.xmax,
        n_grid=                 args.n_grid
    )
    
    gc.collect()
    torch.cuda.empty_cache()


