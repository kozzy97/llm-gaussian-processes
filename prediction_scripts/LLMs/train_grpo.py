import argparse
import re
import numpy as np
import random
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from datasets import Dataset
from unsloth import FastLanguageModel
import torch
import gc
from trl import GRPOConfig, GRPOTrainer
from generate_dataset import gen_gp_dataset
from vllm import SamplingParams
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

def _extract_floats(string: str) -> float:
    pattern = r'[-+]?\d*\.\d+|[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+'
    return [float(match) for match in re.findall(pattern, string)][-1]

def reward_gp_absolute(prompts, completions, ground_truth, **kwargs) -> list[float]:
    # question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r, g in zip(responses, ground_truth):
        try:
            abs_diff = abs(_extract_floats(r) - float(g))
            rewards.append(max(-10.0, -abs_diff))
        except ValueError:
            rewards.append(-11.0)
        except IndexError:
            rewards.append(-11.0)
    return rewards
        
def reward_gp_log_likelihood(prompts, completions, ground_truth, **kwargs) -> list[float]:
    np.random.seed(1997)
    random.seed(1997)
    
    question = prompts[0][-1]["content"].rstrip("-") # get the question, strip any trailing '-'
    try:
        xs, ys = [list(t) for t in zip(*[(float(x), float(y)) for x, y in re.findall(r"X:\s*([\d.]+),\s*Y:\s*([\d.]+)", question)])] # extract the xs and ys
        first_example = False
    except ValueError:
        first_example = True
    last_x = float(re.search(r"X:\s*([\d.]+),\s*Y:\s*$", question).group(1)) # get the x the model must predict for
    
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    ls = 8
    noise = 0.001
    kern_fit = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=ls, length_scale_bounds="fixed") # fix the kernel to be as close to the data generator as possible
    gp = GaussianProcessRegressor( # initialize the GP
                        kernel=kern_fit,
                        alpha=(noise**2), # Assume this level of noise, won't be exactly as in the dataset generation.
                        optimizer=None,
                        normalize_y=False,
                        copy_X_train=True,
                        random_state=1997
                    )
    for r, g in zip(responses, ground_truth): # for each response
        try:
            y_pred = _extract_floats(r) # extract the LLMs prediction
            if not first_example:
                gp.fit(np.array(xs).reshape(-1,1), np.array(ys).ravel()) # fit the GP to the sampled data
            mu, std = gp.predict(np.array(last_x).reshape(-1, 1), return_std=True) # predict at the test point
            var = float(std**2) + noise**2 # observational variance (sum of sigma_f and sigma_eps)
            log_likelihood = float(norm.logpdf(y_pred, loc=mu, scale=np.sqrt(var))) # compute the log likelihood of the ground truth
            rewards.append(max(-999, log_likelihood)) # the log likelihood can be very large negative, cap it
        except ValueError:
            rewards.append(-1000.0)
        except IndexError:
            rewards.append(-1000.0)
    return rewards

def main(seed: int,
        length_scales: list[int] | int,
        num_noise_values: int,
        noise_level_mean: float,
        noise_level_sd: float,
        model_name: str,
        output_dir: str,
        reward_func: str,
        functions_per_noise: int = 100,
        n_samples: int = 50,
        xmin: float = 0.0,
        xmax: float = 29.0,
        n_grid: int = 900) -> None:
    assert reward_func in ["absolute", "log_likelihood"], "Invalid reward function specified. Must be 'absolute' or 'log_likelihood'."
    if reward_func == "log_likelihood":
        assert seed == 1997, "For log_likelihood reward, seed must be 1997."
        assert length_scales == [8], "For log_likelihood reward, length_scales must be [8]."
    reward_gp = reward_gp_absolute if reward_func == "absolute" else reward_gp_log_likelihood

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 3000,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = 32,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    dataset = Dataset.from_list(gen_gp_dataset(seed, 
                                               length_scales, 
                                               num_noise_values, 
                                               noise_level_mean, 
                                               noise_level_sd, 
                                               functions_per_noise=functions_per_noise, 
                                               n_samples=n_samples, 
                                               xmin=xmin, 
                                               xmax=xmax, 
                                               n_grid=n_grid))
    dataset = dataset.map(lambda x: {
                            "prompt" : [
                                {"role": "user",   "content": x["prompt"]},
                            ],
                            "answer": x['ground_truth'],
                        })
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = 32,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = seed,
    )
    vllm_sampling_params = SamplingParams(
            min_p = 0.1,
            top_p = 1.0,
            top_k = -1,
            seed = 3407,
            stop = [tokenizer.eos_token],
            include_stop_str_in_output = True,
        )
    training_args = GRPOConfig(
                vllm_sampling_params = vllm_sampling_params,
                temperature = 1.0,
                learning_rate = 5e-6,
                weight_decay = 0.01,
                warmup_ratio = 0.1,
                lr_scheduler_type = "linear",
                optim = "adamw_8bit",
                logging_steps = 1,
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 1, # Increase to 4 for smoother training
                num_generations = 4, # Decrease if out of memory
                max_prompt_length = 256,
                max_completion_length = 50,
                max_steps = 10000,
                save_steps = 500,
                report_to = "none", # Can use Weights & Biases
                output_dir = output_dir,
            )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_gp,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()    
    
    argparser = argparse.ArgumentParser(description="Train GRPO on a GP dataset.")
    argparser.add_argument("--seed", type=int, required=True, help="Random seed for dataset generation.")
    argparser.add_argument("--length_scales", type=int, nargs='+', required=True, help="Length scales for the GP kernel.")
    argparser.add_argument("--num_noise_values", type=int, required=True, help="Number of noise values to sample.")
    argparser.add_argument("--noise_level_mean", type=float, required=True, help="Mean of the noise level.")
    argparser.add_argument("--noise_level_sd", type=float, required=True, help="Standard deviation of the noise level.")
    argparser.add_argument("--model_name", type=str, required=True, help="Model name for GRPO.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model.")
    argparser.add_argument("--reward_func", type=str, required=True, choices=["absolute", "log_likelihood"], help="Reward function to use: 'absolute' or 'log_likelihood'.")
    argparser.add_argument("--functions_per_noise", type=int, default=100, help="Number of functions per noise level.")
    argparser.add_argument("--n_samples", type=int, default=50, help="Number of samples to draw from the GP.")
    argparser.add_argument("--xmin", type=float, default=0.0, help="Minimum x value for the GP.")
    argparser.add_argument("--xmax", type=float, default=29.0, help="Maximum x value for the GP.")
    argparser.add_argument("--n_grid", type=int, default=900, help="Number of grid points for the GP.")
    args = argparser.parse_args()
    main(
        seed=args.seed,
        length_scales=args.length_scales,
        num_noise_values=args.num_noise_values,
        noise_level_mean=args.noise_level_mean,
        noise_level_sd=args.noise_level_sd,
        model_name=args.model_name,
        output_dir=args.output_dir,
        reward_func=args.reward_func,
        functions_per_noise=args.functions_per_noise,
        n_samples=args.n_samples,
        xmin=args.xmin,
        xmax=args.xmax,
        n_grid=args.n_grid
    )


    gc.collect()
    torch.cuda.empty_cache()


