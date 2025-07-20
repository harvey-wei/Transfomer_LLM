import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        '''
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len), GT integer token IDs
        Caveat: we must subtract the max value of logits before applying softmax for numerical stability
        and cancel out log and exp whenever possible
        '''
        # Subtract the max value of logits
        max_val, _ = torch.max(logits, dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        logits -= max_val # (batch_size, seq_len, vocab_size)

        # log of sum of exp of logits
        log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1)) # (batch_size, seq_len)

        # negative log likelihood
        # batch_idx = torch.arange(logits.shape[0], device=logits.device)[:, None] # (batch_size, 1)
        # seq_idx = torch.arange(logits.shape[1], device=logits.device)[None, :] # (1, seq_len)
        # advanced indexing add complexity to support various dimensionality.
        # use gather to get the logits of the targets. logits and targets have the same number of dimensions
        neg_log_likelihood = log_sum_exp - torch.gather(logits, dim=-1, index=targets[..., None]).squeeze(-1) # (batch_size, seq_len)

        return torch.mean(neg_log_likelihood)