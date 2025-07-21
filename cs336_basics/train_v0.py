import numpy as np
import random
import os
import math
from tqdm import tqdm
import time
import hydra
from hydra.utils import to_absolute_path
import json
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Pool, cpu_count
import wandb
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.ce_loss import CrossEntropyLoss
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.bpe_tokenzier import train_bpe, BPETokenizer
from cs336_basics.transformer_arch import TransformerLM
from cs336_basics.inference import compute_perplexity


# TODO: Use torch.utils.data.DataLoader to load data into batches

def load_tokenized_data_as_numpy_array(data_path: str):
    '''
    Load tokenized data from a file and return it as a numpy array.
    data_path: str
    Returns:
        np.array: tokenized data
    '''
    # token_array behaves like a normal numpy array
    # torch.from_numpy() does not allocate new memory. It shares memory with the NumPy array whenever possible.
    # requires numpy array is writable
    # Hence, set mmap_mode='r+' means read-write mode.
    token_array = np.load(data_path, mmap_mode='r+')

    return token_array

def extract_data_batch_by_iteration(data: np.array, batch_size: int, context_length: int, device: str, iteration: int = 0):
    '''
    Extract a batch of data from a numpy array by start_id.
    data: (token_count, ) tokenized data x = (x_1, x_2, ..., x_token_count)
    start_id: int
    context_length: int
    iteration: int, zero-based index of training iteration.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            input_tokens: (batch_size, context_length)
            next_tokens: (batch_size, context_length)
        e.g. batch_size = 1, ([x_1, x_2, ..., x_context_length], [x_2, x_3, ..., x_context_length + 1])
    n = len(data), m = context_length
    0, 1, ....,n - m,  n-1
    all possible next_tokens start id: 1, 2, ..., n - m
    all possible input_tokens start id: 0, 1, ..., n - m - 1
    We have n - m pairs of input_tokens and next_tokens

    Now we study the start_idx of input_tokens
    iteration is zero-based, i
    batch index is zero-based, b = 0, 1, ..., batch_size - 1

    input_start_idx ith iteration, bth batch: s_{i, b} = i * batch_size + b, b = 0, 1, ..., batch_size - 1
    corresponding end_idx: s_{i, b} + context_length - 1 = i * batch_size + b + context_length - 1
    max end_idx = s_{i, B - 1} = i * batch_size + (B - 1) + context_length - 1 <= len(data) - 2
    '''
    assert iteration * batch_size + (batch_size -1) + context_length <= len(data) - 1, f'?{iteration + 1} * {batch_size} * {context_length} is larger than the number of tokens {len(data)}.'
    input_start_ids = [iteration * batch_size + b for b in range(batch_size)]
    
    input_tokens = [torch.from_numpy(data[start_idx:start_idx + context_length]) for start_idx in input_start_ids]
    next_tokens = [torch.from_numpy(data[start_idx + 1:start_idx + context_length + 1]) for start_idx in input_start_ids]

    input_tokens = torch.stack(input_tokens) # (batch_size, context_length)
    next_tokens = torch.stack(next_tokens) # (batch_size, context_length)

    return (input_tokens.to(device), next_tokens.to(device))

def sample_data_batch(data: np.array, batch_size: int, context_length: int, device: str):
    '''
    Load data into batches of size batch_size and context_length.
    data: (token_count, ) tokenized data x = (x_1, x_2, ..., x_token_count)
    batch_size: int
    context_length: int
    device: str 'cpu' or 'cuda'

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            input_tokens: (batch_size, context_length)
            next_tokens: (batch_size, context_length)
        e.g. batch_size = 1, ([x_1, x_2, ..., x_context_length], [x_2, x_3, ..., x_context_length + 1])
    n = len(data), m = context_length
    0, 1, ....,n - m,  n-1
    all possible next_tokens start id: 1, 2, ..., n - m
    all possible input_tokens start id: 0, 1, ..., n - m - 1
    We have n - m pairs of input_tokens and next_tokens
    '''
    total_pairs = len(data) - context_length
    
    assert total_pairs >= batch_size, 'batch_size is larger than the number of pairs.'

    # Random generate batch_size start_ids of input_tokens from [0, 1, 2, ..., len(data) - context_length - 1]
    # In np.random.randint and torch.randint, the upper bound is exclusive but can not remove duplicates
    # np.random.choice make sures no two start_ids are the same by setting repalce (==put back) to False
    # input_token_start_ids = np.random.randint(0, len(data) - batch_size, batch_size, ) # (batch_size, )
    legal_input_token_start_ids = np.arange(total_pairs)
    input_token_start_ids = np.random.choice(legal_input_token_start_ids, batch_size, replace=False) # (batch_size, )

    input_tokens = [torch.from_numpy(data[start_idx:start_idx + context_length]) for start_idx in input_token_start_ids] # (batch_size, context_length)
    next_tokens = [torch.from_numpy(data[start_idx + 1:start_idx + context_length + 1]) for start_idx in input_token_start_ids] # (batch_size, context_length)

    # stack the list of tensors into a single tensor, dim=0
    input_tokens = torch.stack(input_tokens)
    next_tokens = torch.stack(next_tokens)

    return (input_tokens.to(device), next_tokens.to(device))


@hydra.main(config_path="config", config_name="config.yaml")
def train(cfg: DictConfig):
    # print the config
    print(f"Config: {cfg}")

    # initialize wandb, log hyperparameters and start a new run
    run =  wandb.init(project="cs336-basics", config=OmegaConf.to_container(cfg, resolve=True))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set datatypehttps://chatgpt.com/c/6869e6b5-2cdc-8005-94e6-0cd6f6bcc42d
    # dtype = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float32
    dtype = torch.float32

    # load data as numpy array using mmap
    train_data = load_tokenized_data_as_numpy_array(to_absolute_path(cfg.data.train_path))
    val_data = load_tokenized_data_as_numpy_array(to_absolute_path(cfg.data.val_path))

    # for debugging
    # train_data = train_data[:257]
    # val_data = val_data[:1000]

    print(f'Train data size in tokens: {len(train_data)}') # 543,838,077 tokens
    print(f'Val data size in tokens: {len(val_data)}') # 5,492,351 tokens

    total_iterations_per_epoch = (len(train_data) - cfg.model.context_length) // cfg.training.batch_size
    total_iterations = cfg.training.num_epochs * total_iterations_per_epoch
    print(f'Total iterations per epoch: {total_iterations_per_epoch}')
    print(f'Total iterations: {total_iterations}')

    batch_size = cfg.training.batch_size
    context_length = cfg.model.context_length

    # create checkpoint directory if not exists
    # checkpoint_dir = to_absolute_path(cfg.checkpointing.save_dir)
    checkpoint_dir = cfg.checkpointing.save_dir

    if Path(checkpoint_dir).exists():
        print(f'Checkpoint directory {checkpoint_dir} already exists')
    else:
        print(f'Checkpoint directory {checkpoint_dir} does not exist, creating...')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load vacab from json file
    with open(to_absolute_path(cfg.data.vocab_path), 'r') as f:
        vocab = json.load(f)
    print(f'Vocab size: {len(vocab)}')

    # initialize model
    model = TransformerLM(
        vocab_size=cfg.data.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
        device=device,
        dtype=dtype,
    )
    model.to(device)
    # model.train()

    # log model, gradients, and metrics
    run.watch(model, log="all")

    print(f'Model initialized on {device} with dtype {dtype}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    total_steps_suggested = int(20.0 * (sum(p.numel() for p in model.parameters())) / (cfg.training.batch_size * cfg.model.context_length))
    print(f'Total steps suggested: {total_steps_suggested}')
    print(f'Total steps configured: {cfg.training.max_steps}')

    # return
    
    # initialize loss function
    loss_fn = CrossEntropyLoss()

    # initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        betas=(cfg.training.beta_1, cfg.training.beta_2),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )

    # initialize learning rate scheduler
    cos_lr_min=cfg.training.lr_min_fraction * cfg.training.lr_max
    cos_lr_max=cfg.training.lr_max
    cos_T_warmup=cfg.training.T_warmup
    cos_T_cosine=cfg.training.T_cosine

    print(f"Starting training for {cfg.training.num_epochs} epochs")

    # training loop
    iteration = 0
    # for epoch in tqdm(range(cfg.training.num_epochs), desc=f"Epochs"):
    for epoch in range(cfg.training.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        print("-" * 10)

        # set model to train mode
        model.train()

        # initialize tqdm progress bar
        # pbar = tqdm(range(0, len(train_data) - context_length, batch_size), desc="Iterations")
        # for i in tqdm(range(0, len(train_data) - context_length, batch_size), desc="Iterations"):
        # pbar = tqdm.tqdm(total=iter_cnt_per_epoch, desc="Iterations")
        # all legal input_tokens start idx: 0, 1, 2...., len(train_data) - context_length - 1
        # pbar = tqdm(range(0, len(train_data) - context_length, batch_size), desc="Iterations")
        # start idx of all batches: 0, batch_size, 2 * batch_size, ..., len(train_data) - context_length - 1
        indices = list(range(0, len(train_data) - context_length, batch_size))
        random.shuffle(indices) # shuffle the indices in-place
        pbar = tqdm(indices, desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        # for i in tqdm(indices, desc=f"Iteration {iteration + 1}/{len(indices)}"):
        for i in pbar:
            # check if the last input_token seq of size context_length out of bound
            # if i + batch_size - 1 + context_length - 1 > len(train_data) - 2:
            if i + batch_size + context_length > len(train_data):
                continue

            input_start_ids = [i + b for b in range(batch_size)]

            # The following can be done with process pool 
            input_tokens = [torch.from_numpy(train_data[start_id:start_id + context_length]) for start_id in input_start_ids]
            next_tokens = [torch.from_numpy(train_data[start_id + 1:start_id + context_length + 1]) for start_id in input_start_ids]

            input_tokens = torch.stack(input_tokens).to(device) # (batch_size, context_length)
            next_tokens = torch.stack(next_tokens).to(device) # (batch_size, context_length)
            
        #with torch.autograd.detect_anomaly():
            # forward pass
            logits = model(input_tokens)
            loss = loss_fn(logits, next_tokens)

            # backward pass
            optimizer.zero_grad()

            loss.backward()

            # gradient clipping
            gradient_clipping(model.parameters(), cfg.training.max_grad_l2_norm)

            # updata learning rate
            lr = lr_cosine_schedule(
                t=iteration,
                lr_min=cos_lr_min,
                lr_max=cos_lr_max,
                T_warmup=cos_T_warmup,
                T_cosine=cos_T_cosine,
            )

            # update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # update parameters
            optimizer.step()

            if iteration > 0 and iteration % cfg.checkpointing.save_ckpt_interval == 0:
                # evaluate model on validation data
                # print(f"Evaluating model on validation data at iteration {iteration}")
                # model.eval()
                # with torch.no_grad():
                #     # val_loss = evaluate(model, val_data, cfg.model.context_length, device)
                #     perplexity = compute_perplexity(model, val_data, cfg.model.context_length, cfg.training.batch_size, device)
                #     run.log({
                #         "val/perplexity": perplexity,
                #         "training iteration": iteration,
                #     })
                # model.train()

                # log loss and training iteration
                # run.log({
                #     "train/loss": loss.item(),
                #     "training iteration": iteration,
                # })

                # save checkpoint as file name: checkpoint_time_epoch_iteration.
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                ckpt_file_name = f'ckpt_{current_time}_epoch_{epoch}_iter_{iteration}.pt'
                ckpt_full_path = Path(checkpoint_dir) / ckpt_file_name
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    out=ckpt_full_path,
                )

            run.log({
                 f"train/loss_lr_max_{cos_lr_max:.6f}": loss,
                   f"train/lr_lr_max_{cos_lr_max:.6f}": lr
                   }, step=iteration)


            # increase iteration
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

            iteration += 1

            if iteration >= cfg.training.max_steps:
                # save checkpoint as file name: checkpoint_time_epoch_iteration.
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                ckpt_file_name = f'ckpt_{current_time}_epoch_{epoch}_last_iter.pt'
                ckpt_full_path = Path(checkpoint_dir) / ckpt_file_name
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    out=ckpt_full_path,
                )
                print(f"Reached maximum number of steps: {cfg.training.max_steps}")
                break

if __name__ == "__main__":
    train()