import torch
import torch.nn.functional as F

def sample(model, context, num_tokens, dataset, context_length, temperature=1.0, device="cpu"):
    context = context.to(device)
    generated_tokens = context
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # sequence length is last dimension
            context = torch.cat([context, next_token], dim=-1)
            if context.size()[1] > context_length:
                context = context[:, -context_length:]
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
    
    generated_tokens = generated_tokens[0].cpu().tolist()
    generated_text = None
    generated_text = dataset.detokenize(generated_tokens)
        
    return generated_tokens, generated_text
