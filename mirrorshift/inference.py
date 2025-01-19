"""
This file contains inference-related functions.

References (code):
- https://github.com/huggingface/transformers: for efficient top-p
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    # keep the most probable tokens such that the probs sum to top_p
    if top_p >= 1.0:
        return probs
    values, indices = torch.sort(probs, descending=True)
    cumulative_sums = torch.cumsum(values, dim=-1)
    sorted_indices_to_remove = cumulative_sums <= (1 - top_p)
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=indices,
        src=sorted_indices_to_remove
    )
    output_probs = probs.clone()
    output_probs[indices_to_remove] = 0.0
    return output_probs

def min_p_filter(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    # keep everything that has probability at least
    # most probable token * min_p
    if min_p >= 1.0:
        return probs
    max_prob = torch.max(probs, dim=-1).values.item()
    threshold = max_prob * min_p
    return torch.where(probs >= threshold, probs, 0.0)

def sample(model: torch.nn.Module, context: torch.Tensor, num_tokens: int, subset: Subset, context_length: int, temperature: float=1.0, top_p: float=0.95, min_p: float=0.1, device: str="cpu"):
    context = context.to(device)
    generated_tokens = context

    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # top-p and min-p filtering
            probs = top_p_filter(probs, top_p)
            probs = min_p_filter(probs, min_p)
            next_token = torch.multinomial(probs, num_samples=1)
            # sequence length is last dimension
            context = torch.cat([context, next_token], dim=-1)
            if context.size()[1] > context_length:
                context = context[:, -context_length:]
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

    generated_tokens = generated_tokens[0].cpu().tolist()
    generated_text = None
    generated_text = subset.dataset.detokenize(generated_tokens)

    return generated_tokens, generated_text
