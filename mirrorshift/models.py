"""
This file contains the model definitions.

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

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import ModelConfig

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    # theta variable is base theta, 10000 in original paper
    # theta ^ -2(i-1)/d
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    # imo more readable than ones_like
    freqs_cis = torch.polar(torch.ones(freqs.size()), freqs)
    # output dimensions are (end, dim//2)
    return freqs_cis

class GroupedQueryAttention(nn.Module):
    # for grouped query attention, many queries are grouped together with a single key and value matrix
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        context_length: int,
        attention_dropout_p: float,
        residual_dropout_p: float
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.per_head_dim = self.embedding_dim // self.num_heads
        self.context_length = context_length

        # W_Q, W_K, W_V for all attention heads
        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(
            embedding_dim, self.num_kv_heads * self.per_head_dim, bias=False
        )
        self.V = nn.Linear(
            embedding_dim, self.num_kv_heads * self.per_head_dim, bias=False
        )
        self.output_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.output_proj.weight)

        self.attention_dropout_p = attention_dropout_p
        self.residual_dropout = nn.Dropout(p=residual_dropout_p)

    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_heads, per_head_dim = x.size()
        # reshaping does the "division into d/2" subspaces
        # viewed as complex so you can simply multiply the complex number
        # x+iy by the precomputed cos theta + i sin theta, rotating vector (x,y) by theta
        x_complex = torch.view_as_complex(
            x.reshape(batch_size, seq_length, num_heads, per_head_dim // 2, 2)
        )
        freqs_cis = freqs_cis[:seq_length].view(
            1, seq_length, 1, per_head_dim // 2
        )  # flatten for multiply
        x_rotated = x_complex * freqs_cis
        # reshape back to normal
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch_size, seq_length, num_heads, per_head_dim)
        return x_out

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
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
        V = V.view(
            batch_size, seq_length, self.num_kv_heads, self.per_head_dim
        ).transpose(1, 2)
        Q = F.rms_norm(Q, (Q.size(-1),))
        K = F.rms_norm(K, (K.size(-1),))
        Q = self.apply_rope(Q, freqs_cis).transpose(1, 2)
        K = self.apply_rope(K, freqs_cis).transpose(1, 2)

        # calling efficient attention kernel
        output = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            dropout_p=self.attention_dropout_p,
            is_causal=True,
            enable_gqa=True,
        )

        # transpose to move num_heads back to original dim and then recombine
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.embedding_dim)
        )
        output = self.output_proj(output)
        output = self.residual_dropout(output)
        return output

class MultiHeadLatentAttention(nn.Module):
    # for MLA, queries, keys, and values are all projected down to a low-rank latent tensor
    # before being up-projected for the attention mechanism
    # one latent for queries and a joint latent for keys and values
    # this lets the joint latents be cached instead of the entire key and value pair
    # the reason head_dims are specified here is because deepseek tends to use
    # head_dim values greater than embedding_dim / num_heads
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        context_length: int,
        attention_dropout_p: float,
        residual_dropout_p: float,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim

        self.context_length = context_length

        # W_Q, W_K, W_V for all attention heads
        self.Q_a = nn.Linear(self.embedding_dim, self.q_lora_rank, bias=False)
        # note, num_heads * qk_head_dim does not have to be equal to embedding_dim
        self.Q_b = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # joint down-projection to kv_lora as well as decoupled part of key for rope
        # this could be two matrices but it's done jointly
        self.KV_a = nn.Linear(
            self.embedding_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        # upscales for the nope part of the key and for the value
        # this could be two matrices but it's done jointly
        self.KV_b = nn.Linear(
            self.kv_lora_rank,
            self.num_kv_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )
        self.output_proj = nn.Linear(
            self.num_heads * self.v_head_dim, embedding_dim, bias=False
        )
        nn.init.zeros_(self.output_proj.weight)

        self.attention_dropout_p = attention_dropout_p
        self.residual_dropout = nn.Dropout(p=residual_dropout_p)

    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_heads, per_head_dim = x.size()
        # reshaping does the "division into d/2" subspaces
        # viewed as complex so you can simply multiply the complex number
        # x+iy by the precomputed cos theta + i sin theta, rotating vector (x,y) by theta
        x_complex = torch.view_as_complex(
            x.reshape(batch_size, seq_length, num_heads, per_head_dim // 2, 2)
        )
        freqs_cis = freqs_cis[:seq_length].view(
            1, seq_length, 1, per_head_dim // 2
        )  # flatten for multiply
        x_rotated = x_complex * freqs_cis
        # reshape back to normal
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch_size, seq_length, num_heads, per_head_dim)
        return x_out

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        Q = self.Q_a(x)
        Q = F.rms_norm(Q, (Q.size(-1),))  # low rank is normalized
        Q = self.Q_b(Q)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
        Q_nope, Q_rope = torch.split(
            Q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        Q_rope = self.apply_rope(Q_rope, freqs_cis)
        Q = torch.cat([Q_nope, Q_rope], dim=-1).transpose(1, 2)

        # for keys, we want the up-projection for rope to be decoupled
        # since rope is position-sensitive, we can't cache rope-d keys
        # cache is not relevant for this code (yet) but explains why the rope-d
        # part of the keys are decoupled - refer to deepseek paper for more
        kv_latent_plus_rope = self.KV_a(x)
        kv_latent, k_rope = torch.split(
            kv_latent_plus_rope, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # k_rope here is common for all attn heads
        k_rope = k_rope.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        k_rope = k_rope.expand(
            batch_size, seq_length, self.num_kv_heads, self.qk_rope_head_dim
        )
        k_rope = self.apply_rope(k_rope, freqs_cis).transpose(1, 2)

        kv_latent = F.rms_norm(kv_latent, (kv_latent.size(-1),))
        kv = self.KV_b(kv_latent)
        kv = kv.view(
            batch_size,
            seq_length,
            self.num_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # calling efficient attention kernel
        output = F.scaled_dot_product_attention(
            query=Q,
            key=k,
            value=v,
            dropout_p=self.attention_dropout_p,
            is_causal=True,
            enable_gqa=True,
        )

        # transpose to move num_heads back to original dim and then recombine
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.num_heads * self.v_head_dim)
        )
        output = self.output_proj(output)
        output = self.residual_dropout(output)
        return output

class MLP(nn.Module):
    # position-wise FF layer
    def __init__(self, model_dim: int, feedforward_dim: int):
        super().__init__()
        self.to_hidden = nn.Linear(model_dim, feedforward_dim, bias=False)
        self.from_hidden = nn.Linear(feedforward_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dimensions of x are batch_size, seq_length, embedding_dim
        # which is also batch_size, seq_length, model_dim
        return self.from_hidden(F.relu(self.to_hidden(x)).square())

class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig
    ):
        super().__init__()
        if model_config.attention_type == "gqa":
            self.attention_block = GroupedQueryAttention(
            embedding_dim=model_config.embedding_dim,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            context_length=model_config.context_length,
            attention_dropout_p=model_config.attention_dropout_p,
            residual_dropout_p=model_config.residual_dropout_p,
            )
        elif model_config.attention_type == 'mla':
            self.attention_block = MultiHeadLatentAttention(
                embedding_dim=model_config.embedding_dim,
                num_heads=model_config.num_heads,
                num_kv_heads=model_config.num_kv_heads,
                q_lora_rank=model_config.q_lora_rank,
                kv_lora_rank=model_config.kv_lora_rank,
                qk_nope_head_dim=model_config.qk_nope_head_dim,
                qk_rope_head_dim=model_config.qk_rope_head_dim,
                v_head_dim=model_config.v_head_dim,
                context_length=model_config.context_length,
                attention_dropout_p=model_config.attention_dropout_p,
                residual_dropout_p=model_config.residual_dropout_p,
            )
        else:
            raise ValueError(f"Unknown attention type: {model_config.attention_type}")
        self.ff_block = MLP(
            model_dim=model_config.embedding_dim, feedforward_dim=model_config.feedforward_dim
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # should be straightforward, dimensions stay the same
        output = F.rms_norm(x, (x.size(-1),))  # last dim is embedding_dim
        output = output + self.attention_block(output, freqs_cis)
        output = F.rms_norm(output, (output.size(-1),))  # last dim is embedding_dim
        output = output + self.ff_block(output)
        return output

class CausalTransformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig
    ):
        # vocab size is equal for input and output
        super().__init__()
        self.embedding_layer = nn.Embedding(model_config.vocab_size, model_config.embedding_dim)
        if model_config.attention_type == 'gqa':
            freqs_cis = precompute_freqs_cis(model_config.embedding_dim // model_config.num_heads, model_config.context_length)
        elif model_config.attention_type == 'mla':
            freqs_cis = precompute_freqs_cis(model_config.qk_rope_head_dim, model_config.context_length)
        else:
            raise ValueError(f"Unknown attention type: {model_config.attention_type}")
        self.register_buffer("freqs_cis", freqs_cis)

        self.decoder_stack = nn.ModuleList(
            [
                DecoderBlock(model_config = model_config)
                for _ in range(model_config.num_layers)
            ]
        )
        self.lm_head = nn.Linear(model_config.embedding_dim, model_config.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is batch size, seq_len where each value is a token index
        output = self.embedding_layer(x)
        output = F.rms_norm(output, (output.size(-1),))
        for layer in self.decoder_stack:
            output = layer(output, self.freqs_cis)
        output = F.rms_norm(output, (output.size(-1),))
        output = self.lm_head(output)
        return output
