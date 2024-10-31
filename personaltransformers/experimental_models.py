from models import MultiHeadAttention_Fast, PositionalEncoding

import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadFFN(nn.Module):
    # position-wise FF layer - each head gets its own nonlinear FFN
    # then the heads are mixed
    def __init__(self, model_dim, feedforward_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0
        assert feedforward_dim % num_heads == 0

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.per_head_dim = model_dim // num_heads
        self.per_head_hidden_dim = feedforward_dim // num_heads

        self.to_hidden = nn.Parameter(torch.randn(num_heads, self.per_head_dim, self.per_head_hidden_dim))
        self.from_hidden = nn.Parameter(torch.randn(num_heads, self.per_head_hidden_dim, self.per_head_dim))
        nn.init.kaiming_uniform_(self.to_hidden, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.from_hidden, nonlinearity='linear')

        self.mixing = nn.Linear(self.model_dim, self.model_dim)

    def forward(self, x):
        # dimensions of x are batch_size, seq_length, embedding_dim
        # which is also batch_size, seq_length, model_dim
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.per_head_dim)
        hidden = torch.einsum('bshd,hdk->bshk', x, self.to_hidden)
        hidden = F.relu(hidden)
        output = torch.einsum('bshk,hkd->bshd', hidden, self.from_hidden)
        output = output.reshape(batch_size, seq_length, self.model_dim).contiguous()
        output = self.mixing(output)
        return output

class DecoderBlockMHFFN(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, feedforward_dim, attention_dropout_p, residual_dropout_p):
        super().__init__()
        self.attention_block = MultiHeadAttention_Fast(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            attention_dropout_p=attention_dropout_p,
            residual_dropout_p=residual_dropout_p
        )
        self.layer_norm_attention = nn.LayerNorm(embedding_dim)
        self.ff_block = MultiHeadFFN(
            model_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads
        )
        self.layer_norm_ff = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # should be straightforward, dimensions stay the same
        output = self.layer_norm_attention(x)
        output = output + self.attention_block(output)
        output = self.layer_norm_ff(output)
        output = output + self.ff_block(output)
        return output

class DecoderOnlyTransformerMHFFN(nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_dim, num_heads, context_length, feedforward_dim, attention_dropout_p, residual_dropout_p):
        # vocab size is equal for input and output
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding_layer = PositionalEncoding(context_length, embedding_dim)
        self.decoder_stack = nn.ModuleList([
            DecoderBlockMHFFN(
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
        return output

