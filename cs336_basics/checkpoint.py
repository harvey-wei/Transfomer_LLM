import torch
import torch.nn as nn
import os
import typing


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int, 
                    out= str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    print("=> Saving checkpoint")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration
        }
        , out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer):
    print("=> Loading checkpoint")
    ckpt = torch.load(src)  # ckpt is the object saved by torch.save()
    iteration = ckpt["iteration"]
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    return iteration
