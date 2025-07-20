import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat
import numpy as np
import math
# from bpe_tokenzier import BPETokenizer


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, bias=False):
        '''
        Linear layer is a point-wise linear transformation which means it does not depends on batch size, sequence length.
        Only depends on the input features and output features dimensions
        
        Attention layer is a point-wise linear transformation which means it does not depends on batch size, sequence length.
        Only depends on the input features and output features dimensions

        RMSNorm layer is a point-wise linear transformation which means it does not depends on batch size, sequence length.
        Only depends on the input features and output features dimensions

        In contrast, Embedding layer is a sequence-wise linear transformation which means it depends on batch size, sequence length.
        It is a lookup table for the embedding vectors.

        SwiGLU is also a point-wise linear transformation which means it does not depends on batch size, sequence length.
        Only depends on the input features and output features dimensions

        Rope is not a linear transformation which means it depends on the sequence length and the input features dimensions.

        Transfomer block consists of multiple layers, including Linear, RMSNorm, SwiGLU, Rope, and MultiheadAttention.
        Hence it is a point-wise linear transformation which means it does not depends on batch size, sequence length.

        Convolutional layer is also a point-wise linear transformation which means it does not depends on batch size, sequence length.
        Only depends on the input features and output features dimensions
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.initialize()
    
    def initialize(self):
        var = 2 / (self.in_features + self.out_features)
        std_dev = math.sqrt(var)
        nn.init.trunc_normal_(self.weight, mean=0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        ''' 
        x: (batch_size, seq_len, in_features)
        '''
        x = einsum(x, self.weight, 'batch seq_len in_features, out_features in_features -> batch seq_len out_features')
        if self.bias is not None:
            x = x + self.bias
        return x

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim, device: torch.device | None = None, dtype: torch.dtype | None = None):
        ''' 
        num_embeddings: int, Size of the vocabulary
        embedding_dim: int, Size of the embedding vectors, i.e.d_model
        device: torch.device | None, Device to store the parameters
        dtype: torch.dtype | None, Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype), requires_grad=True)
        self.initialize()

    def initialize(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.LongTensor):
        ''' 
        Perform the lookup for the embedding vectors.
        x: (batch_size, seq_len) of Torch.LongTensor, the integer token ids to be looked up in the embedding matrix
        '''
        return self.weight[x, :]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        ''' 
        RMSNorm is a point-wise linear transformation which means it does not depends on batch size, sequence length.
        d_model: int, hidden dimension of the model

        There is a d_model-dimensional scale vector, which is a learnable parameter.
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(1, 1,d_model, device=device, dtype=dtype, requires_grad=True))
        self.initialize()

    def initialize(self):
        nn.init.ones_(self.gain)

    def forward(self, x: torch.Tensor):
        ''' 
        x: (batch_size, seq_len, d_model)
        '''
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) # (batch_size, seq_len, 1)
        # x = x * self.gain[None, None, :] / rms
        # x = x * self.gain[None, None, :] / rms
        x = x * self.gain / rms
    
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        ''' 
        SwiGLU is a point-wise linear transformation which means it does not depends on batch size, sequence length.
        d_model: int, hidden dimension of the model

        SwiGLU(x, w1, w2, w3) = w2(SiLU(w1x) * (w3x))
        SiLU(x) = x * sigmoid(x)

        x is R^d_model
        w1, w2, w3 are learnable parameters.
        w1: (d_ff, d_model)
        w2: (d_model, d_ff)
        w3: (d_ff, d_model)
        '''
        super().__init__()
        self.d_model = d_model
        # self.d_ff = 8 * d_model / 3
        self.d_ff = d_ff

        # assert self.d_ff % 64 == 0, "d_ff must be a mutiple of 64 for hardware efficiency"

        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype, requires_grad=True))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype, requires_grad=True))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype, requires_grad=True))

        self.initialize()   
    
    def initialize_linear(self, weight: torch.Tensor):
        var = 2 / (weight.shape[0] + weight.shape[1])
        std_dev = math.sqrt(var)
        nn.init.trunc_normal_(weight, mean=0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
    
    def initialize(self):
        for weight in [self.w1, self.w2, self.w3]:
            self.initialize_linear(weight)
    
    def forward(self, x: torch.Tensor):
        ''' 
        x: (batch_size, seq_len, d_model)
        '''
        x1 = einsum(self.w1, x, 'd_ff d_model, batch seq_len d_model -> batch seq_len d_ff')
        x1 = x1 * F.sigmoid(x1)

        x3 = einsum(self.w3, x, 'd_ff d_model, batch seq_len d_model -> batch seq_len d_ff')
        
        x2 = x1 * x3

        return einsum(self.w2, x2, 'd_model d_ff, batch seq_len d_ff -> batch seq_len d_model')

class RoPE(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        ''' 
        RoPE is not a linear transformation which means it depends on the sequence length and the input features dimensions.
        theta: float, the frequency scaling factor
        d_k: int, the dimension of the key and query
        max_seq_len: int, the maximum sequence length

        By use max_seq_len, we can precompute the RoPE matrix, which is a (max_seq_len, d_k) matrix.
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        assert d_k % 2 == 0, "d_k must be a mutiple of 2"

        half_dims = torch.arange(self.d_k // 2, device=self.device) #[d_k // 2, ]
        freqs = torch.pow(self.theta, -2 * half_dims / self.d_k)[None, :] # [1, d_k // 2]

        pos_ids = torch.arange(self.max_seq_len, device=self.device)[:, None] # [max_seqlen, 1]
        
        # calcuate the freq for each pair of positon and dim of half_dims by broadcasting
        '''
        When performing operations on tensors with different shapes:

        Align shapes from right to left.

        Two dimensions are compatible if:

        They are equal, or

        One of them is 1

        Missing (leading) dimensions are treated as 1

        If all dimensions are compatible, broadcasting occurs by expanding size-1 dims

        Note that broadcasting is virtual and no new memory is allocated.
        It eanbles efficient vectorization

        '''
        thetas_pos_dim = pos_ids * freqs # [max_seq_len, d_k //2]

        # sin and cos computation is expensive, we compute once for forward use and register as buffers
        # for inference

        self.sin_theta_i_k = torch.sin(thetas_pos_dim) # [max_seq_len, d_k // 2]
        self.cos_theta_i_k = torch.cos(thetas_pos_dim)


        self.register_buffer('rope_cos_theta_i_k', self.cos_theta_i_k)
        self.register_buffer('rope_sin_theta_i_k', self.sin_theta_i_k)


        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        '''
        x: (batch_size,seq_len, d_model)
        token_positions: (batch_size, seq_len)
        Efficient implementation at reference: https://arxiv.org/pdf/2104.09864#page=4.75
        We treat head_num as batch dim
        '''
        batch_size, seq_len, d_model = x.shape

        # print(f"x.device: {x.device}, self.device: {self.device}")
        # print(f"cos_theta_i_k.device: {self.cos_theta_i_k.device}, sin_theta_i_k.device: {self.sin_theta_i_k.device}")

        if x.device != self.device:
            x = x.to(self.device)
            token_positions = token_positions.to(self.device)

        if token_positions.ndim == 1:
            token_positions = einops.repeat(token_positions, "seq_len -> batch_size seq_len",
                                            batch_size= batch_size)
        
        assert token_positions.shape == (batch_size, seq_len), \
                f"token_positions shape {token_positions.shape}! But it must be a ({batch_size}, {seq_len}) tensor"

        out = torch.empty_like(x) # (batch_size, seq_len, d_model)
        out[..., 0::2] = self.cos_theta_i_k[token_positions, :] * x[..., 0::2] - \
                self.sin_theta_i_k[token_positions, :] * x[..., 1::2]
        out[..., 1::2] = self.sin_theta_i_k[token_positions, :] * x[..., 0::2] + \
                self.cos_theta_i_k[token_positions, :] * x[..., 1::2]

        return out


class RoPEV1(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        RoPE is not a linear transformation which means it depends on the sequence length and the input features dimensions.
        theta: float, the frequency scaling factor
        d_k: int, the dimension of the key and query
        max_seq_len: int, the maximum sequence length

        By use max_seq_len, we can precompute the RoPE matrix, which is a (max_seq_len, d_k) matrix.
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        assert d_k % 2 == 0, "d_k must be a mutiple of 2"

        # Here, these two tensors requires no graidents, are not part of state_dict
        self.cos_theta_i_d = torch.empty(max_seq_len, d_k, device=device, dtype=dtype, requires_grad=False)
        self.sin_theta_i_d = torch.empty(max_seq_len, d_k, device=device, dtype=dtype, requires_grad=False)

        self.compute_theta_i_k_matrix()

        # By registering cos_theta_i_d and sin_theta_i_d as buffer,
        # they are part of state_dict, if presistent = True
        # auto-moved by .to(device), .cuda()
        # not to be trained no gradients, no optimizer updates
        # Here, we set persistent to False such that these two tensor buffers are not saved in
        # in state_dict() but can be recomputed during inference.
        self.register_buffer('cos_theta_i_k', self.cos_theta_i_d, persistent=False)
        self.register_buffer('sin_theta_i_k', self.sin_theta_i_d, persistent=False)

    def compute_theta_i_k_matrix(self):
        '''
        Compute the theta_i_k matrix, which is a (max_seq_len, d_k) matrix.
        theta_i_k = theta * i / d_k
        in orignal paper both and k index start from 1
        '''
        # original implementation for grasping the idea
        # for i in range(self.max_seq_len):
        #     for k in range(self.d_k // 2):
        #         theta_i_k = i * (self.theta ** (-2 * k / self.d_k))
        #         self.cos_theta_i_d[i, 2 * k] = math.cos(theta_i_k)
        #         self.sin_theta_i_d[i, 2 * k] = math.sin(theta_i_k)

        #         self.cos_theta_i_d[i, 2 * k + 1] = self.cos_theta_i_d[i, 2 * k]
        #         self.sin_theta_i_d[i, 2 * k + 1] = self.sin_theta_i_d[i, 2 * k]
        
        # Vectorize the computation of cos and sin
        freq = self.theta ** (-2 * torch.arange(self.d_k // 2) / self.d_k)[None, ] # (1, d_k // 2,)
        freq = freq.to(self.device)
        seq_len_i = torch.arange(self.max_seq_len, device=self.device, dtype=self.dtype)[:, None] # (max_seq_len, 1)
        theta_i_k = seq_len_i * freq # (max_seq_len, d_k // 2)

        self.cos_theta_i_d[:, 0::2] = torch.cos(theta_i_k)
        self.sin_theta_i_d[:, 0::2] = torch.sin(theta_i_k)
        self.cos_theta_i_d[:, 1::2] = self.cos_theta_i_d[:, 0::2]
        self.sin_theta_i_d[:, 1::2] = self.sin_theta_i_d[:, 0::2]
        

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        '''
        x: (batch_size,seq_len, d_model)
        token_positions: (batch_size, seq_len)
        Efficient implementation at reference: https://arxiv.org/pdf/2104.09864#page=4.75
        We treat head_num as batch dim
        '''
        batch_size, seq_len, d_model = x.shape

        # original implementation
        # Premature optimization is the root of all evil!
        # for i in range(seq_len):
        #     x_i = torch.empty([batch_size, head_num, d_model], device=x.device, dtype=x.dtype)
        #     x_i[:,..., 0::2] = -x[:, ..., i, 1::2] # (batch_size, d_model // 2)
        #     x_i[:,..., 1::2] = x[:,..., i, 0::2] # (batch_size, d_model // 2)

        #     x[:, ..., i, :] = x[:, ..., i, :] * self.cos_theta_i_d[token_positions[i], :] + x_i * self.sin_theta_i_d[token_positions[i], :]
        # end of original implementation

        if token_positions.ndim == 1:
            token_positions = token_positions[None, :]
            token_positions = token_positions.expand(batch_size, seq_len)
        
        assert token_positions.shape == (batch_size, seq_len), f"token_positions shape {token_positions.shape}! But it must be a ({batch_size}, {seq_len}) tensor"

        # # efficient implementation according to the original paper
        # x_i = torch.empty([batch_size, seq_len, d_model], device=x.device, dtype=x.dtype)
        # x_i[..., 0::2] = -x[..., 1::2]
        # x_i[..., 1::2] = x[..., 0::2]
        #
        # # Use torch.tensor indexing array instead of slicing
        # x = x * self.cos_theta_i_d[token_positions, :] + x_i * self.sin_theta_i_d[token_positions, :]

        # efficient implementation by odd-even indexing
        y = torch.empty_like(x)
        y[:, :, 0::2] = x[..., 0::2] * self.cos_theta_i_d[token_positions, 0::2] - \
                x[..., 1::2] * self.sin_theta_i_d[token_positions, 0::2]
        y[:, :, 1::2] = x[..., 1::2] * self.cos_theta_i_d[token_positions, 1::2] + \
                x[..., 0::2] * self.sin_theta_i_d[token_positions, 1::2]

        return y

def softmax_safe(input: torch.Tensor, i: int):
    '''
    input: (A, B, C, ..) any shape
    i: int, the index of the dimension to apply softmax
    '''
    # assert i < input.ndim, "i must be less than the number of dimensions of input"
    max_val, max_idx_dim_i = torch.max(input, dim=i, keepdim=True)
    input = input - max_val
    exp_input = torch.exp(input)

    return exp_input / torch.sum(exp_input, dim=i, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    '''
    Q: (batch_size, seq_len_q, d_k)
    K: (batch_size, seq_len_k, d_k)
    V: (batch_size, seq_len_k, d_v)
    mask: (seq_len_q, seq_len_k) with mask[i, j] = True if q_i attends to k_j
    '''
    d_k = Q.shape[-1]

    if mask is not None:
        mask = torch.where(mask, 0.0, float('-inf'))
    else:
        # no head_num, mask is None
        mask = torch.zeros_like([Q.shape[0], Q.shape[1], K.shape[1]], dtype=Q.dtype, device=Q.device)
    
    if mask.ndim == 2:
        mask = mask[None, :, :]
    
    # Note that d_k is a python scaler so you have to use math.sqrt(d_k) not torch.sqrt(d_k)
    # torch.sqrt(torch.Tensor(d_k, dtype=Q.dtype)) is a workaround to avoid type error
    # sim_score = einsum(Q, K, "batch seq_len_q d_k, batch seq_len_k d_k -> batch seq_len_q seq_len_k") / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    # ... means any number of dimensions like head_num
    sim_score = einsum(Q, K, "batch ... seq_len_q d_k, batch ... seq_len_k d_k -> batch ... seq_len_q seq_len_k") / math.sqrt(d_k)

    # exp(0) = 1 attend
    # exp(-inf) = 0 not attend
    sim_score = softmax_safe(sim_score + mask, i=-1)

    return einsum(sim_score, V, "batch ... seq_len_q seq_len_k, batch ... seq_len_k d_v -> batch ... seq_len_q d_v")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None, rope: RoPE | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype) 
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        '''
        x: (batch_size, seq_len, d_model)
        theta: float, RoPE parameter
        max_seq_len: int, maximum sequence length
        token_positions: torch.Tensor shape[1, seq_len], token positions

        Causal Multihead Self-Attention
        Apply RoPE to the query and key

        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],


        '''
        x = x.to(self.device)
        batch_size, seq_len, d_model = x.shape

        # Project the input to query, key, value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Rearrange the query, key, value to (batch_size * num_heads, seq_len, head_dim)
        q = rearrange(q, "batch seq_len (num_heads head_dim) -> (batch num_heads) seq_len head_dim", head_dim=self.head_dim)
        k = rearrange(k, "batch seq_len (num_heads head_dim) -> (batch num_heads) seq_len head_dim", head_dim=self.head_dim)
        v = rearrange(v, "batch seq_len (num_heads head_dim) -> (batch num_heads) seq_len head_dim", head_dim=self.head_dim)

        # RoPE the query and key
        # if theta is not None and max_seq_len is not None and token_positions is not None:
        if self.rope is not None and token_positions is not None:
            token_positions = token_positions.to(self.device)
            # rope = RoPE(theta, self.head_dim, max_seq_len, device=self.device, dtype=self.dtype)

            # We assume token_positions is a (1, seq_len) tensor
            token_positions = repeat(token_positions,  "single seq_len -> (batch single num_heads) seq_len", batch=batch_size, num_heads=self.num_heads)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # Extract the lower triangular part (includes diagonal) of the causal mask -> 1 (truthy)
        # The other part is 0
        causal_mask = torch.tril(torch.full((seq_len, seq_len), True, device=self.device, dtype=torch.bool))
        
        x = self._scaled_dot_product_attention(q, k, v, causal_mask) # (batch_size * num_heads, seq_len, head_dim)

        x = rearrange(x, "(batch num_heads) seq_len head_dim -> batch seq_len (num_heads head_dim)", num_heads=self.num_heads)

        x = self.out_proj(x)

        return x

    @staticmethod
    def _softmax_safe(input: torch.Tensor, i: int):
        '''
        input: (A, B, C, ..) any shape
        i: int, the index of the dimension to apply softmax
        '''
        assert i < input.ndim, "i must be less than the number of dimensions of input"
        max_val, max_idx_dim_i = torch.max(input, dim=i, keepdim=True)
        input = input - max_val
        exp_input = torch.exp(input)

        return exp_input / torch.sum(exp_input, dim=i, keepdim=True)

    @staticmethod
    def _scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        '''
        Q: (batch_size, seq_len_q, d_k)
        K: (batch_size, seq_len_k, d_k)
        V: (batch_size, seq_len_k, d_v)
        mask: (seq_len_q, seq_len_k) with mask[i, j] = True if q_i attends to k_j
        '''
        d_k = Q.shape[-1]

        if mask is not None:
            mask = torch.where(mask, 0.0, float('-inf'))
        else:
            # no head_num, mask is None
            mask = torch.zeros_like([Q.shape[0], Q.shape[1], K.shape[1]], dtype=Q.dtype, device=Q.device)
        
        if mask.ndim == 2:
            mask = mask[None, :, :]
        
        # Note that d_k is a python scaler so you have to use math.sqrt(d_k) not torch.sqrt(d_k)
        # torch.sqrt(torch.Tensor(d_k, dtype=Q.dtype)) is a workaround to avoid type error
        # sim_score = einsum(Q, K, "batch seq_len_q d_k, batch seq_len_k d_k -> batch seq_len_q seq_len_k") / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
        # ... means any number of dimensions like head_num
        sim_score = einsum(Q, K, "batch ... seq_len_q d_k, batch ... seq_len_k d_k -> batch ... seq_len_q seq_len_k") / math.sqrt(d_k)

        # exp(0) = 1 attend
        # exp(-inf) = 0 not attend
        sim_score = MultiHeadAttention._softmax_safe(sim_score + mask, i=-1)

        return einsum(sim_score, V, "batch ... seq_len_q seq_len_k, batch ... seq_len_k d_v -> batch ... seq_len_q d_v")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None, rope: RoPE | None = None):
        '''
        x -> RMSNorm -> MultiheadAttention -> RMSNorm -> SwiGLU -> y
        d_model -> MultiheadAttention -> d_model -> SwiGLU -> d_model
        in SwiGLU: d_ff = 8 * d_model / 3. d_model -> d_ff -> d_model

        Here, we only use one RoPE instance for the whole transformer-based model.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.rope = rope

        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, device=device, dtype=dtype, rope=rope)
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        '''
        x: (batch_size, seq_len, d_model)
        token_positions: (1, seq_len)
        '''
        batch_size, seq_len, d_model = x.shape

        if token_positions is not None:
            assert token_positions.ndim == 2, "token_positions must be a (1, seq_len) tensor"
            assert token_positions.shape[0] == 1, "token_positions must be a (1, seq_len) tensor"
            assert token_positions.shape[1] == seq_len, "token_positions must be a (1, seq_len) tensor"
        
        x1 = self.rms_norm_1(x)
        x1 = self.multihead_attn(x1, token_positions)
        x = x + x1

        x2 = self.rms_norm_2(x)
        x2 = self.swiglu(x2)
        x = x + x2

        return x


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        '''
        input_seq -> Tokenizer -> in_indices -> Embedding -> in_embeddings -> [TransformerBlock] x num_layers -> Norm -> Linear -> logits -> softmax -> PDF over Vocab (-> Tokenizer -> output_seq)
        vocab embedding matrix is of shape (vocab_size, d_model)
        RoPE's max_seq_len is context_length
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype

        # self.tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=special_tokens)
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device, dtype=dtype)

        # Sequential or ModuleList?
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, rope=self.rope) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor):
        '''
        x: (batch_size, seq_len), batch_size x seq_len of integer token IDs
        '''
        batch_size, seq_len = x.shape
        if seq_len > self.context_length:
            x = x[:, :self.context_length]
            seq_len = self.context_length
        token_positions = torch.arange(start=0, end=seq_len, step=1, device=self.device, dtype=torch.long)[None, :]

        x = self.embedding(x) # [batch_size, seq_len, d_model]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, token_positions)
        
        x = self.norm(x)
        x = self.lm_head(x) # [batch_size, seq_len, vocab_size] logits are not normalized probability

        # x = softmax_safe(x, i=-1)
        # # x = F.softmax(x, dim=-1)

        return x
    
    @staticmethod
    def _softmax_safe(input: torch.Tensor, i: int):
        '''
        input: (A, B, C, ..) any shape
        i: int, the index of the dimension to apply softmax
        '''
        max_val, _ = torch.max(input, dim=i, keepdim=True)
        input = input - max_val
        exp_input = torch.exp(input)

        return exp_input / torch.sum(exp_input, dim=i, keepdim=True)

def test_swiglu():
    swiglu = SwiGLU(d_model=128 * 8 // 3)
    x = torch.randn(1, 1024, 128 * 8 // 3)
    desired_shape = x.shape
    assert swiglu(x).shape == desired_shape
    print("SwiGLU test passed")


def test_rms_norm():
    rms_norm = RMSNorm(d_model=3)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    desired_shape = x.shape
    assert rms_norm(x).shape == desired_shape
    print("RMSNorm test passed")


def test_embedding():
    embedding = Embedding(num_embeddings=10, embedding_dim=3)
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])
    desired_shape = x.shape + (embedding.embedding_dim,)
    assert embedding(x).shape == desired_shape
    print("Embedding test passed")

if __name__ == "__main__":
    # test_embedding()
    # test_rms_norm()
    # test_swiglu()

    d_k = 128
    d_k = torch.tensor(d_k, dtype=torch.float32)
    print(math.sqrt(d_k))
