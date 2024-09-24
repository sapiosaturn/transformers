"""
References (code):
- https://github.com/karpathy/minGPT
- https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse

References (papers)
- Attention is all you need: https://arxiv.org/pdf/1706.03762
- DeepSeek-V2: https://arxiv.org/pdf/2405.04434
- GQA: https://arxiv.org/pdf/2305.13245

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE: for now, these implmentations do not use FlashAttention, and are slow
# NOTE: decoder-only transformers for now, so no 

class MultiHeadAttention(nn.Module):
    # TODO: add bias?
    def __init__(
        self,
        embedding_dim: int, 
        num_heads: int, 
        attention_dropout: float, 
        residual_dropout: float
        ):

        super().__init__()
        assert embedding_dim % num_heads == 0 

        # W_Q, W_K, W_V for all attention heads
        # from attention is all you need:
        # "For each of these we use d_k = d_v = d_model/h = 64"
        # embedding_dim used twice because 
        self.attention_matrices = nn.Linear(embedding_dim, 3*embedding_dim)
        # self.register_buffer("causal_mask", causal_mask)

class GroupedQueryAttention(nn.Module):
    pass

class MultiHeadLatentAttention(nn.Module):
    pass

