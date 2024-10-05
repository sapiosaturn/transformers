"""
References (code):
- https://github.com/karpathy/minGPT
- https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- https://pytorch.org/tutorials/beginner/translation_transformer.html
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse

References (papers)
- Attention is all you need: https://arxiv.org/pdf/1706.03762
- DeepSeek-V2: https://arxiv.org/pdf/2405.04434
- GQA: https://arxiv.org/pdf/2305.13245

"""

import math

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
        output = self.residual_dropout(output) # is this really a "residual" dropout?
        return output

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

class PositionalEncoding(nn.Module):
    # TODO: dropout?
    def __init__(self, context_length, model_dim): # model_dim = embedding_dim
        super().__init__()
        positional_encoding = torch.zeros((context_length, model_dim))
        denominator = torch.exp(- torch.arange(0, model_dim, 2)* math.log(10000) / model_dim)
        pos = torch.arange(0, context_length).reshape(context_length, 1)
        positional_encoding[:, 0::2] = torch.sin(pos * denominator)
        positional_encoding[:, 1::2] = torch.cos(pos * denominator)
        positional_encoding = positional_encoding.reshape(1, context_length, model_dim)
        # final dimensions of positional encoding: 
        # batch_size (after being broadcasted onto input, context_length, model_dim
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        _, seq_length, _ = x.size()
        return x + self.positional_encoding[:, :seq_length, :]

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, feedforward_dim, attention_dropout_p, residual_dropout_p):
        super().__init__()
        self.attention_block = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
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

    def forward(self, x):
        # should be straightforward, dimensions stay the same
        output = x + self.attention_block(x)
        output = self.layer_norm_attention(output)
        output = output + self.ff_block(output)
        output = self.layer_norm_ff(output)
        return output


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_dim, num_heads, context_length, feedforward_dim, attention_dropout_p, residual_dropout_p):
        # vocab size is equal for input and output
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding_layer = PositionalEncoding(context_length, embedding_dim)
        self.decoder_stack = nn.ModuleList([
            DecoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
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
        output = self.positional_encoding_layer(output)
        for layer in self.decoder_stack:
            output = layer(output)
        output = self.final_linear(output)
        # for output probabilities
        output = F.softmax(output, dim=-1)
        return output


class GroupedQueryAttention(nn.Module):
    pass

class MultiHeadLatentAttention(nn.Module):
    pass

