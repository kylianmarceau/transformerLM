from __future__ import annotations
import torch
from src.tokenizer import END_OF_TEXT

def sample_next_token(logits,temperature=1.0,top_k=None,top_p=None,generator=None,):
    # choose one token from a batch of logits 
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    if top_k is not None:
        if not 1 <= top_k <= logits.size(-1):
            raise ValueError("top_k must be between 1 and the vocabulary size")

    if top_p is not None:
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

    # 1. Temperature scaling
    filtered_logits = logits / temperature

    # 2. Top-k filtering
    if top_k is not None:
        top_values, top_indices = torch.topk(filtered_logits,top_k,dim=-1,)
        kept_logits = torch.full_like(filtered_logits,float("-inf"),)
        kept_logits.scatter_(dim=-1,index=top_indices,src=top_values,)
        filtered_logits = kept_logits

    # 3. Top-p filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(filtered_logits,descending=True,dim=-1,)
        sorted_probabilities = torch.softmax(sorted_logits,dim=-1,)
        cumulative_probabilities = torch.cumsum(sorted_probabilities,dim=-1,)
        remove = cumulative_probabilities > top_p
        # Shift the mask right so the token that crosses top_p is kept
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove,float("-inf"),)
        filtered_logits = torch.full_like(filtered_logits,float("-inf"),)
        filtered_logits.scatter_(dim=-1,index=sorted_indices,src=sorted_logits,)

    # 4. Renormalise and sample
    probabilities = torch.softmax(filtered_logits,dim=-1,)

    return torch.multinomial(probabilities,num_samples=1,generator=generator,)

@torch.no_grad()
def generate(model,tokenizer,prompt: str,max_new_tokens: int = 256,temperature: float = 1.0,top_k: int | None = None,top_p: float | None = None,seed: int | None = None,use_cache: bool = True,):
    """Generate text from a prompt
    Generation stops at the end of text token, max_new_tokens, or context_length
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError("The prompt must encode to at least one token")

    context_length = model.config.context_length

    if len(prompt_ids) > context_length:
        raise ValueError("The prompt is longer than the model context length")

    device = next(model.parameters()).device

    token_ids = torch.tensor([prompt_ids],dtype=torch.long,device=device,)

    available_positions = context_length - len(prompt_ids)
    number_to_generate = min(max_new_tokens,available_positions,)

    if number_to_generate == 0:
        return tokenizer.decode(token_ids[0].tolist())

    try:
        eot_id = tokenizer.special_to_id[END_OF_TEXT]
    except KeyError as error:
        raise ValueError(
            "The tokenizer does not contain the EOT token"
        ) from error

    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    model.eval()
    model.reset_cache()

    # Prefill the complete prompt
    logits = model(token_ids,use_cache=use_cache,)

    for step in range(number_to_generate):
        next_token = sample_next_token(logits[:, -1],temperature=temperature,top_k=top_k,top_p=top_p,generator=generator,)

        token_ids = torch.cat((token_ids, next_token),dim=1,)

        if next_token.item() == eot_id:
            break

        if step == number_to_generate - 1:
            break

        if use_cache:
            # Decode only the newly generated token
            logits = model(next_token,use_cache=True,)
        else:
            # Rerun the complete sequence without caching
            logits = model(token_ids,use_cache=False,)

    return tokenizer.decode(token_ids[0].tolist())