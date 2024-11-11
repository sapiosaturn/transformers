"""
References (code):
- https://github.com/karpathy/minGPT
- https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- https://pytorch.org/tutorials/beginner/translation_transformer.html
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse
- https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L90 (for RoPE)

References (papers)
- Attention is all you need: https://arxiv.org/pdf/1706.03762
- DeepSeek-V2: https://arxiv.org/pdf/2405.04434
- GQA: https://arxiv.org/pdf/2305.13245

"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE: decoder-only transformers for now, so no custom masks

class GroupedQueryAttention(nn.Module):
    # for grouped query attention, many queries are grouped together with a single key and value matrix
    def __init__(self, embedding_dim, num_heads, num_kv_heads, context_length, attention_dropout_p, residual_dropout_p):
        super().__init__()
        assert embedding_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.per_head_dim = self.embedding_dim // self.num_heads
        self.per_kv_head_queries = self.num_heads // self.num_kv_heads
        self.context_length = context_length

        # W_Q, W_K, W_V for all attention heads
        self.Q = nn.Linear(embedding_dim, embedding_dim)
        self.K = nn.Linear(embedding_dim, self.num_kv_heads * self.per_head_dim)
        self.V = nn.Linear(embedding_dim, self.num_kv_heads * self.per_head_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attention_dropout_p = attention_dropout_p
        self.residual_dropout = nn.Dropout(p = residual_dropout_p)

    def apply_rope(self, x, freqs_cis):
        batch_size, seq_length, num_heads, per_head_dim = x.size()
        # reshaping does the "division into d/2" subspaces
        # viewed as complex so you can simply multiply the complex number
        # x+iy by the precomputed cos theta + i sin theta, rotating vector (x,y) by theta
        x_complex = torch.view_as_complex(
            x.reshape(batch_size, seq_length, num_heads, per_head_dim//2, 2)
        )
        freqs_cis = freqs_cis[:seq_length].view(1, seq_length, 1, per_head_dim//2) # flatten for multiply
        x_rotated = x_complex * freqs_cis
        # reshape back to normal
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch_size, seq_length, num_heads, per_head_dim)
        return x_out

    def forward(self, x, freqs_cis):
        batch_size, seq_length, _ = x.size()
        # each q, k, v matrix is (batch size, seq_length, embedding_dim, embedding_dim) here
        # forward pass on the attention_matrices linear layer results in calculating queries, keys, values
        # corresponding to the tokens that are there
        # transposing here so next operations are done per attention head 
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.per_head_dim)
        K = K.view(batch_size, seq_length, self.num_kv_heads, self.per_head_dim)
        V = V.view(batch_size, seq_length, self.num_kv_heads, self.per_head_dim).transpose(1,2)

        Q = self.apply_rope(Q, freqs_cis).transpose(1,2)
        K = self.apply_rope(K, freqs_cis).transpose(1,2)

        # calling efficient attention kernel
        output = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            dropout_p=self.attention_dropout_p,
            is_causal=True,
            enable_gqa=True
        )

        # transpose to move num_heads back to original dim and then recombine
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        output = self.output_proj(output)
        output = self.residual_dropout(output)
        return output

def precompute_freqs_cis(dim, end, theta = 10000.0):
    # theta variable is base theta, 10000 in original paper
    # theta ^ -2(i-1)/d
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    # imo more readable than ones_like
    freqs_cis = torch.polar(torch.ones(freqs.size()), freqs)
    # output dimensions are (end, dim//2)
    return freqs_cis

class PositionWiseFeedForward(nn.Module):
    # position-wise FF layer
    def __init__(self, model_dim, feedforward_dim):
        super().__init__()
        self.to_hidden = nn.Linear(model_dim, feedforward_dim)
        self.from_hidden = nn.Linear(feedforward_dim, model_dim)

    def forward(self, x):
        # dimensions of x are batch_size, seq_length, embedding_dim
        # which is also batch_size, seq_length, model_dim
        return self.from_hidden(F.relu(self.to_hidden(x)))

class DecoderBlockGQA(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_kv_heads, context_length, feedforward_dim, attention_dropout_p, residual_dropout_p):
        super().__init__()
        self.attention_block = GroupedQueryAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            context_length=context_length,
            attention_dropout_p=attention_dropout_p,
            residual_dropout_p=residual_dropout_p
        )
        self.layer_norm_attention = nn.LayerNorm(embedding_dim)
        self.ff_block = PositionWiseFeedForward(
            model_dim=embedding_dim,
            feedforward_dim=feedforward_dim
        )
        self.layer_norm_ff = nn.LayerNorm(embedding_dim)

    def forward(self, x, freqs_cis):
        # should be straightforward, dimensions stay the same
        output = self.layer_norm_attention(x)
        output = output + self.attention_block(output, freqs_cis)
        output = self.layer_norm_ff(output)
        output = output + self.ff_block(output)
        return output

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, num_layers, num_kv_heads, embedding_dim, num_heads, context_length, feedforward_dim, attention_dropout_p, residual_dropout_p):
        # vocab size is equal for input and output
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        freqs_cis = precompute_freqs_cis(embedding_dim//num_heads, context_length)
        self.register_buffer("freqs_cis", freqs_cis)
        self.decoder_stack = nn.ModuleList([
            DecoderBlockGQA(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                context_length=context_length,
                feedforward_dim=feedforward_dim,
                attention_dropout_p=attention_dropout_p,
                residual_dropout_p=residual_dropout_p
            ) for _ in range(num_layers)
        ])
        self.final_linear = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        # x is batch size, seq_len where each value is a token index
        output = self.embedding_layer(x)
        for layer in self.decoder_stack:
            output = layer(output, self.freqs_cis)
        output = self.final_linear(output)
        # return logits instead of probabilities for sampling flexibility
        return output

class MultiHeadLatentAttention(nn.Module):
    pass

