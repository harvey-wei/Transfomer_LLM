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
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TokenizedDataset(Dataset):
    def __init__(self, data_path: str, context_length: int):
        '''
            Use torch.utils.data.DataLoader to load data into batches
            Load data in format of npy by memmap
        '''
        self.data = np.load(data_path, mmap_mode='r+')
        self.context_length = context_length
        self.total_sample_len = len(self.data) - context_length
    
    def __len__(self):
        '''
        This sets the valid indices for items: 0, 1, 2, ..., len(dataset)-1.
        These indices point to individual samples in the dataset.
        RandomSampler permutes these indices if shuffle=True, the DataLoader automatically uses a RandomSampler.
        '''
        return self.total_sample_len
    
    def __getitem__(self, idx):
        '''
        If your Dataset.__getitem__ returns something like (input_tokens, next_tokens), then
        batch = [ (input_tokens_1, next_tokens_1), (input_tokens_2, next_tokens_2), ..., (input_tokens_B, next_tokens_B) ]
        B = batch_size
        batch is what is given to the collate_fn
        '''
        input_tokens = torch.from_numpy(self.data[idx:idx + self.context_length])
        next_tokens = torch.from_numpy(self.data[idx + 1:idx + self.context_length + 1])
        return input_tokens, next_tokens
    
def token_collate_fn(batch):
    ''' 
    batch is a list of tuples, each tuple is (input_tokens, next_tokens)
    '''
    input, targets = zip(*batch)
    input_tokens = torch.stack(input, dim=0) # (batch_size, context_length)
    next_tokens = torch.stack(targets, dim=0) # (batch_size, context_length)
    # input_tokens = torch.stack([item[0] for item in batch])
    # next_tokens = torch.stack([item[1] for item in batch])

    return input_tokens, next_tokens
 
def build_loader(data_path, context_length, batch_size, num_workers=4, pin_memory=True, shuffle=True):
    dataset = TokenizedDataset(data_path, context_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=token_collate_fn,
    )
    return loader

def test_TokenizedDataset():
    dataset = TokenizedDataset(data_path="data/TinyStoriesV2-GPT4-train_10000.npy", context_length=4)
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
    print(dataset[5])

    inputs, nexts = token_collate_fn([dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]])
    print(inputs)
    print(nexts)

def test_build_loader():
    loader = build_loader(data_path="data/TinyStoriesV2-GPT4-train_10000.npy", context_length=4, batch_size=2, num_workers=2, pin_memory=True, shuffle=True)
    for inputs, nexts in loader:
        print(f"inputs: {inputs}")
        print(f"nexts: {nexts}")
    
if __name__ == "__main__":
    test_TokenizedDataset()
    test_build_loader()