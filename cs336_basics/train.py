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
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.ce_loss import CrossEntropyLoss
from cs336_basics.optimizer import AdamWLR, CosineLR, gradient_clipping
from cs336_basics.bpe_tokenzier import train_bpe, BPETokenizer
from cs336_basics.transformer_arch import TransformerLM
from cs336_basics.inference import compute_perplexity, compute_val_loss
from cs336_basics.data import  build_loader



@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def train(cfg: DictConfig):
    # Hydra creates a fresh run directory and sets cwd to it:
    # e.g., outputs/2025-07-20/XX-XX-XX
    logdir = os.getcwd()
    print(f"TensorBoard logs -> {logdir}")
    writer = SummaryWriter(log_dir=logdir)

    print(f"Config: {cfg}")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set data type
    dtype = torch.float32

    # create dataloader for train and val
    train_loader = build_loader(
            to_absolute_path(cfg.data.train_path),
            cfg.model.context_length,
            cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            shuffle=True
    )
    print(f"Train set size: {len(train_loader.dataset)}")

    val_loader = build_loader(
            to_absolute_path(cfg.data.val_path),
            cfg.model.context_length,
            cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            shuffle=False)
    print(f"Val set size: {len(val_loader.dataset)}")

    # Note we train only one epoch
    print(f'For large dataset, we train only one epoch')
    total_iterations = (len(train_loader.dataset) - cfg.model.context_length) // cfg.training.batch_size
    print(f'Total iterations: {total_iterations}')

    batch_size = cfg.training.batch_size
    context_length = cfg.model.context_length
    max_steps = cfg.training.max_steps

    # create checkpoint directory if not exists
    # checkpoint_dir = to_absolute_path(cfg.checkpointing.save_dir)
    checkpoint_dir = cfg.checkpointing.save_dir

    if Path(checkpoint_dir).exists():
        print(f'Checkpoint directory {checkpoint_dir} already exists')
    else:
        print(f'Checkpoint directory {checkpoint_dir} does not exist, creating...')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model
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
    model.train()

    # log model, gradients, and metrics
    # writer.add_graph(model, input_to_model=torch.randint(0, cfg.data.vocab_size, (1, context_length)))

    # create loss function
    loss_fn = CrossEntropyLoss()

    # create optimizer
    cos_lr_min=cfg.training.lr_min_fraction * cfg.training.lr_max
    cos_lr_max=cfg.training.lr_max
    cos_T_warmup=cfg.training.T_warmup
    cos_T_cosine=cfg.training.T_cosine

    optimizer = AdamWLR(
        model.parameters(),
        lr_schedule=CosineLR(cos_lr_min, cos_lr_max, cos_T_warmup, cos_T_cosine),
        lr=cfg.training.lr_max,
        betas=(cfg.training.beta_1, cfg.training.beta_2),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
        last_step=-1,
    )

    # training loop over data_loader directly with epoch
    step = 0
    pbar = tqdm(train_loader, desc=f"Training step {step} / {max_steps}")
    for input_tokens, next_tokens in pbar:
        # move data to device
        input_tokens = input_tokens.to(device)
        next_tokens = next_tokens.to(device)

        # forward pass
        logits = model(input_tokens)
        loss = loss_fn(logits, next_tokens)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), cfg.training.max_grad_l2_norm)
        optimizer.step()

        # log metrics
        if step > 0 and step % cfg.checkpointing.save_ckpt_interval == 0:
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            ckpt_file_name = f'ckpt_{current_time}_step_{step}.pt'
            ckpt_full_path = Path(checkpoint_dir) / ckpt_file_name
            save_checkpoint(model, optimizer, step, ckpt_full_path)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")


        # logging loss and learning rate
        # if step > 0 and step % cfg.checkpointing.log_interval == 0:
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
        writer.flush()

        # check if training step is over
        step += 1
        if step == max_steps:
            print(f"Training step {step} / {max_steps} is over")
            # save checkpoint
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            ckpt_file_name = f'ckpt_{current_time}_final.pt'
            ckpt_full_path = Path(checkpoint_dir) / ckpt_file_name
            save_checkpoint(model, optimizer, step, ckpt_full_path)
            break

    # close tensorboard writer
    writer.close()
    print(f"TensorBoard logs -> {logdir}")

    # save config
    with open(Path(checkpoint_dir) / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

if __name__ == "__main__":
    train()