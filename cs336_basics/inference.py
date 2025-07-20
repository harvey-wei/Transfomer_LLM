import torch
import math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import random
import yaml
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from cs336_basics.transformer_arch import TransformerLM, softmax_safe
from cs336_basics.bpe_tokenzier import BPETokenizer
from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.ce_loss import CrossEntropyLoss
from cs336_basics.optimizer import CosineLR

from torch.serialization import add_safe_globals
add_safe_globals({"CosineLR": CosineLR})


def get_perplexity(self, logits: torch.Tensor, targets: torch.Tensor):
    '''
    logits: (batch_size=1, seq_len=1, vocab_size)
    targets: (batch_size=1, seq_len=1), GT integer token IDs
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


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    '''
    Sample next token from logits by top-p sampling.
    ref: https://github.com/ari-holtzman/degen/blob/master/gen.py
    logit: (batch_size, vocab_size)
    '''
    assert temperature > 1e-6,  "temperature must be positive"

    # temperature scaled softmax
    probs = softmax_safe(logits.div_(temperature), -1)

    # sort the probs in descending order
    # sorted_indices: (batch_size, vocab_size) tracks the original indices of the probs for tokens
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1) # (batch_size, vocab_size )

    # cumsum the probs
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1) # (batch_size, vocab_size)

    # find the indices where cumsum_probs is less than top_p
    sorted_indcies_to_remove = cumsum_probs > top_p # (batch_size, vocab_size)

    # Trick: we want to keep the first prob > top_p by shift mask to right by 1
    # [0, 0, 1, 1, 1] -> [_, 0, 0, 1, 1] -> [0, 0, 0, 1, 1]
    # extreme case: [0, 0, 0, 1]
    # extreme case: [1, 1, 1, 1]
    # use clone to avoid in-place assignment issues for overlapping memory
    sorted_indcies_to_remove[:, 1:] = sorted_indcies_to_remove[:, :-1].clone()
    sorted_indcies_to_remove[:, 0] = 0

    # set the probs to 0 for the indices to remove
    sorted_probs[sorted_indcies_to_remove] = 0

    # sample from the top-p tokens
    sampled_indices = sorted_probs.multinomial(1).reshape(-1, 1) # (batch_size, 1)
    next_token_indices = sorted_indices.gather(dim=-1, index=sampled_indices) # (batch_size, 1)

    return next_token_indices # (batch_size, 1)

def padding_batch(prompts: list[str], tokenizer: BPETokenizer, device: torch.device = 'cuda') -> torch.Tensor:
    '''
    Pad a batch of prompts to the same length.
    prompts: list[str]

    return: (batch_size, seq_len) tensor of padded token IDs
    seq_len <= max_len
    Note: 0 is the |endoftext| token for padding
    '''
    prompts_tokens = [tokenizer.encode(prompt) for prompt in prompts] # list[list[int]]
    max_prompt_len = max([len(p_t) for p_t in prompts_tokens])

    # pad the prompts to the same length by list concatenation
    prompts_tokens = [p_t + [0] * (max_prompt_len - len(p_t)) for p_t in prompts_tokens] # [batch_size, max_prompt_len]

    # stack the prompts tokens into a tensor
    prompts_tokens_tensor = torch.tensor(prompts_tokens, device=device) # (batch_size, max_prompt_len)

    return prompts_tokens_tensor

def generate_text_batch(model: TransformerLM,
                        tokenizer: BPETokenizer,
                        prompt: list[str],
                        max_tokens: int,
                        temperature: float = 1.0,
                        top_p: float = 0.95,
                        ):
    '''
    Generate text from a model and a tokenizer.
    model: TransformerLM
    tokenizer: BPETokenizer
    prompt: list[str]
    max_tokens: int
    temperature: float
    top_p: float
    Note: 0 is the |endoftext| token
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # pad the prompts to the same length
    prompts_tokens = padding_batch(prompt, tokenizer, device) # (batch_size, max_prompt_len)

    batch_size = prompts_tokens.shape[0]

    # end token is <|endoftext|>
    # end_token = tokenizer.encode("<|endoftext|>")[0] # int
    end_token = tokenizer.vocab_reverse[b"<|endoftext|>"] # int
    # print(f"end_token: {end_token}")

    # generate text
    curr_tokens = prompts_tokens # (batch_size, max_prompt_len)

    next_token_ids = [[] for _ in range(batch_size)] # list[list[int]] list of next batch of  token ids
    # is_ith_batch_done = [False for _ in range(batch_size)] # list[bool] list of whether the ith batch is done
    # is_ith_batch_done = np.zeros(batch_size, dtype=np.int32) # (batch_size,)
    is_ith_batch_done = torch.full((batch_size,), False, dtype=torch.bool, device=device) # (batch_size,)

    with torch.no_grad():
        # while next_token == end_token and  generated_token_cnt < max_tokens:
        for _ in range(max_tokens):
            # forward pass
            logits = model(curr_tokens) # (batch_size, seq_len, vocab_size)

            # sample next token by taking the last token logit only
            # note logits[:, -1, :] is (batch_size, vocab_size) due to -1 indexing not slicing!
            # logits[:, 1:2, :] is of shape (batch_size, 1, vocab_size)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_p) # (batch_size, 1)

            next_token = next_token.squeeze(-1) # (batch_size,)

            # update the batch mask
            end_token_mask = (next_token == end_token) # (batch_size,)
            is_ith_batch_done = is_ith_batch_done | end_token_mask # (batch_size,)
            next_token = next_token.unsqueeze(-1) # (batch_size, 1)

            # pad fisnied batch to end_token
            curr_tokens[is_ith_batch_done] = end_token

            # note that we maintain the batch_size in the loop to avoid reallocation of memory
            # use is_ith_batch_done to track the batch that is done
            # for i in range(batch_size):
            #     if next_token[i, 0].item() == end_token:
            #         is_ith_batch_done[i] = 1

            # print(f"is_ith_batch_done: {is_ith_batch_done}")

            # append next token to current tokens
            for i in range(batch_size):
                if not is_ith_batch_done[i]:
                    next_token_ids[i].append(next_token[i, 0].item())

            # update curr_tokens for next iteration
            curr_tokens = torch.cat([curr_tokens, next_token], dim=-1) # (batch_size, curr_len + 1)
        
        return next_token_ids

def generate_text(model: TransformerLM,
                  tokenizer: BPETokenizer,
                  prompt: str,
                  max_tokens: int,
                  temperature: float = 1.0,
                  top_p: float = 0.95,
                  ):
    '''
    Generate text from a model and a tokenizer.
    model: TransformerLM
    tokenizer: BPETokenizer
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # tokenize the prompt into sequence of token IDs
    prompt_tokens = tokenizer.encode(prompt)  # list[int], (seq_len, )
    prompt_tokens = torch.tensor(prompt_tokens, device=device) # (seq_len, )
    prompt_tokens = prompt_tokens[None, :] # (1, seq_len)

    # TODO: take as input a batch of prompt tokens and padding to same sequence length

    # end token is <|endoftext|>
    end_token = tokenizer.encode("<|endoftext|>")[0] # int

    # generate text
    curr_tokens = prompt_tokens
    next_token_ids = [] # list[int] list of next token ids
    next_token = None

    with torch.no_grad():
        # while next_token == end_token and  generated_token_cnt < max_tokens:
        for _ in range(max_tokens):
            # forward pass
            logits = model(curr_tokens) # (1, seq_len, vocab_size)

            # sample next token by taking the last token logit only
            # note logits[:, -1, :] is (batch_size = 1, vocab_size) due to -1 indexing not slicing!
            # logits[:, 1:2, :] is of shape (batch_size = 1, 1, vocab_size)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_p) # (batch_size = 1, 1)
            # check if next token is end token assume batch_size = 1
            if next_token[0, 0] == end_token:
                break

            # append next token to current tokens
            next_token_ids.append(next_token[0, 0])

            # update curr_tokens for next iteration
            curr_tokens = torch.cat([curr_tokens, next_token], dim=-1) # (batch_size = 1, seq_len + 1)
        
        return next_token_ids


def compute_perplexity(model, token_data: np.ndarray, context_length: int, batch_size: int, device: str = 'cuda'):
    """
    Compute per-token perplexity for a language model on given tokenized data.

    Args:
        model: PyTorch autoregressive LM (returns logits)
        token_data: numpy array of token ids (shape: [N])
        context_length: number of input tokens used to predict next tokens
        batch_size: how many sequences per batch
        device: 'cpu' or 'cuda'
    
    Returns:
        perplexity (float)
    """
    model.eval()
    model.to(device)

    nll_total = 0.0
    count = 0

    with torch.no_grad():
        # iterate through all possible input token start indices
        # i is the start index of input tokens for ith batch i = 0, 1,  
        # For ith batch, the jth seq [i + j, i + j + context_length) is the input tokens, j = 0, 1, ... batch_size - 1
        # Last input seq in ith batch is [i + batch_size - 1, i + batch_size - 1 + context_length)
        # Last target seq in ith batch is [i + batch_size, i + batch_size + context_length)
        for i in range(0, len(token_data) - context_length, batch_size):
            # Ensure we don't exceed end
            if i + batch_size + context_length > len(token_data):
                break

            # Build input and target batches
            input_batch = [torch.from_numpy(token_data[j : j + context_length]) for j in range(i, i + batch_size)]
            target_batch = [torch.from_numpy(token_data[j + 1 : j + 1 + context_length]) for j in range(i, i + batch_size)]

            input_batch = torch.stack(input_batch).to(device)     # (B, context)
            target_batch = torch.stack(target_batch).to(device)   # (B, context)

            logits = model(input_batch)  # (B, context, vocab)
            log_probs = F.log_softmax(logits, dim=-1) # (B, context, vocab)
            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),  # (B * context, vocab)
                target_batch.view(-1),  # (B * context)
                reduction='sum'  # total NLL over all tokens
            )

            nll_total += loss.item()
            count += target_batch.numel()  # number of tokens = B * context
    
    return math.exp(nll_total / count)

    # ppl = torch.exp(torch.tensor(nll_total / count))
    # return ppl.item()

def compute_val_loss(config_path: str, checkpoint_path: str, val_data_path: str):
    '''
    Compute validation loss for a language model on given tokenized data.
    config_path: str, path to config.yaml saved by hydra
    checkpoint_path: str, path to checkpoint.pt saved by torch.save
    val_data_path: str, path to val_data.npy saved by np.save
    '''
    # load config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = DictConfig(config)
    
    context_length = config.model.context_length
    batch_size = config.training.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct the model according to config
    model = TransformerLM(
        vocab_size=config.data.vocab_size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta,
        device=device,
    )

    # load checkpoint after transform path to absolute path
    ckpt = torch.load(to_absolute_path(checkpoint_path))  # ckpt is the object saved by torch.save()
    model.load_state_dict(ckpt["model"])

    # load val data
    val_data = np.load(to_absolute_path(val_data_path), mmap_mode='r+')


    indices = list(range(0, len(val_data) - config.model.context_length, config.training.batch_size))
    random.shuffle(indices) # shuffle the indices in-place
    pbar = tqdm(indices, desc=f"Compute validation loss")

    loss_fn = CrossEntropyLoss()
    loss_sum = 0.0
    sample_cnt = 0

    with torch.no_grad():
                # for i in tqdm(indices, desc=f"Iteration {iteration + 1}/{len(indices)}"):
        for i in pbar:
            # check if the last input_token seq of size context_length out of bound
            # if i + batch_size - 1 + context_length - 1 > len(train_data) - 2:
            if i + batch_size + context_length > len(val_data):
                continue

            input_start_ids = [i + b for b in range(batch_size)]

            # The following can be done with process pool 
            input_tokens = [torch.from_numpy(val_data[start_id:start_id + context_length]) for start_id in input_start_ids]
            next_tokens = [torch.from_numpy(val_data[start_id + 1:start_id + context_length + 1]) for start_id in input_start_ids]

            input_tokens = torch.stack(input_tokens).to(device) # (batch_size, context_length)
            next_tokens = torch.stack(next_tokens).to(device) # (batch_size, context_length)
            
        #with torch.autograd.detect_anomaly():
            # forward pass
            logits = model(input_tokens)
            loss = loss_fn(logits, next_tokens)
            loss = loss.item() * batch_size * context_length
            loss_sum += loss
            sample_cnt += batch_size * context_length
    
    val_loss = loss_sum / sample_cnt
    return val_loss



def test_compute_perplexity():
    model = TransformerLM(
        vocab_size=10000,
        context_length=100,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    token_data = np.random.randint(0, 10000, (9999,))
    context_length = 100
    batch_size = 10
    ppl = compute_perplexity(model, token_data, context_length, batch_size)
    print(f"Perplexity: {ppl}")

def test_padding_batch():
    prompts = ["Hello, world!", "This is a test for list of prompts"]
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="data/vocab_tiny_stories_train_10000.json",
        merges_filepath="data/merges_tiny_stories_val_10000.json",
    )
    prompts_tokens = padding_batch(prompts, tokenizer)
    print(prompts_tokens)

def test_generate_text_batch():
    model = TransformerLM(
        vocab_size=10000,
        context_length=100,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    tokenizer = BPETokenizer.from_files(
        vocab_filepath="data/vocab_tiny_stories_train_10000.json",
        merges_filepath="data/merges_tiny_stories_val_10000.json",
    )

    prompts = ["Hello, world!", "This is a test for list of prompts"]
    # prompt = "Hello, world!"
    # next_token_ids = generate_text(model, tokenizer, prompt, max_tokens=10)

    next_token_ids = generate_text_batch(model, tokenizer, prompts, max_tokens=10)
    print(next_token_ids)

def test_F_nll_loss():
    log_probs = torch.tensor([
        [-0.2, -1.5, -2.0],  # sample 0
        [-2.3, -0.1, -3.0]   # sample 1
    ])
    targets = torch.tensor([0, 1])

    loss = F.nll_loss(log_probs, targets)
    print(loss)


    logits = torch.tensor([
        [0.2, 1.5, 2.0],  # sample 0
        [2.3, 0.1, 3.0]   # sample 1
    ])
    log_probs_2 = F.log_softmax(logits, dim=-1)

    # F.nll_loss(log(softmax(logits, dim=1)),targets) ==  F.cross_entropy(logits,targets)
    loss_2_ce = F.cross_entropy(logits, targets)
    loss_2_nll = F.nll_loss(log_probs_2, targets)
    print(loss_2_ce)
    print(loss_2_nll)


if __name__ == "__main__":
    # test_padding_batch()
    # test_generate_text_batch()
    # test_compute_perplexity()
    # test_F_nll_loss()

    # config_path = "outputs/2025-07-20/00-49-37/.hydra/config.yaml"
    # checkpoint_path = "outputs/2025-07-20/00-49-37/checkpoints/ckpt_2025-07-20_02-50-30_epoch_0_iter_19800.pt"

    config_path = "outputs/2025-07-20/19-12-02/checkpoints/config.yaml"
    checkpoint_path = "outputs/2025-07-20/19-12-02/checkpoints/ckpt_2025-07-20_20-53-41_final.pt"

    val_loss = compute_val_loss(config_path, checkpoint_path, "data/TinyStoriesV2-GPT4-valid_10000.npy")

    # save the val loss to a file
    with open("outputs/2025-07-20/00-49-37/val_loss.txt", "w") as f:
        f.write(f"Val loss: {val_loss}")

    print(f"Val loss: {val_loss}")
