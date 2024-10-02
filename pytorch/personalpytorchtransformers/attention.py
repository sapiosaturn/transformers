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
    # mostly taken from minGPT
    # TODO: add bias?
    # TODO: add parameterized attention matrix sizes
    def __init__(self, embedding_dim, num_heads, context_length, attention_dropout_p, residual_dropout_p):
        super().__init__()
        assert embedding_dim % num_heads == 0 

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.per_head_dim = self.embedding_dim // self.num_heads
        self.context_length = context_length

        # W_Q, W_K, W_V for all attention heads
        # embedding_dim used twice because 
        # from attention is all you need:
        # "For each of these we use d_k = d_v = d_model/h = 64
        self.attention_matrices = nn.Linear(embedding_dim, 3*embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        # causal mask for stopping tokens from attending to tokens after them
        # dimensions are 1 here for broadcasting
        causal_mask = torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length)
        self.register_buffer("causal_mask", causal_mask)

        self.attention_dropout = nn.Dropout(p = attention_dropout_p)
        self.residual_dropout = nn.Dropout(p = residual_dropout_p)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        # each q, k, v matrix is (batch size, seq_length, embedding_dim, embedding_dim) here
        # forward pass on the attention_matrices linear layer results in calculating queries, keys, values
        # corresponding to the tokens that are there
        Q, K, V = self.attention_matrices(x).split(self.embedding_dim, dim=2)
        # transposing here so next operations are done per attention head 
        Q = Q.view(batch_size, seq_length, self.num_heads, self.per_head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.per_head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.per_head_dim).transpose(1,2)
        # Q*K^T / sqrt(d_k)
        attention_scores = torch.einsum('...ik,...jk->...ij', Q, K) / (K.size(-1)**0.5)
        # causal mask since we're doing decoder-only transformers
        attention_scores = attention_scores.masked_fill(self.causal_mask[:, :, :seq_length, :seq_length] == 0, float('-inf'))
        # TODO: wasn't there a sigmoid attention paper recently? try that as well instead of softmax
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)
        # attention_scores * V
        output = torch.einsum('...ik,...kj->...ij', attention_scores, V)
        # transpose to move num_heads back to original dim and then recombine
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        output = self.residual_dropout(output)
        return output

class GroupedQueryAttention(nn.Module):
    pass

class MultiHeadLatentAttention(nn.Module):
    pass

