# save as train_tb.py
import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def train(cfg: DictConfig):
    # Hydra creates a fresh run directory and sets cwd to it:
    # e.g., outputs/2025-07-20/XX-XX-XX
    logdir = os.getcwd()
    print(f"TensorBoard logs -> {logdir}")
    writer = SummaryWriter(log_dir=logdir)

    # Dummy training loop
    for step in range(10):
        loss = torch.sin(torch.tensor(step * 0.1)) + torch.rand(1) * 0.1
        writer.add_scalar("train/loss", loss.item(), global_step=step)
    writer.close()

if __name__ == "__main__":
    train()