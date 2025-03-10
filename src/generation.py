import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import add_gumbel_noise, get_num_transfer_tokens

def compute_combined_logits(model, x, delta, block_indices, score_model, score_weight, cfg_scale, mask_id):
    """
    Compute model logits with optional classifier-free guidance and add delta.
    """
    if cfg_scale > 0.:
        prompt_index = (x != mask_id)
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_cat = torch.cat([x, un_x], dim=0)
        logits_out = model(x_cat).logits
        logits_main, un_logits = torch.chunk(logits_out, 2, dim=0)
        main_logits = un_logits + (cfg_scale + 1) * (logits_main - un_logits)
    else:
        main_logits = model(x).logits

    scorer_logits = score_model(x).logits if score_model is not None else main_logits.detach()
    combined_logits = main_logits.clone()
    combined_logits[:, block_indices] += (score_weight * scorer_logits[:, block_indices] + delta[:, block_indices])
    return combined_logits, scorer_logits

def gradient_update_step(combined_logits, scorer_logits, block_indices):
    """
    Compute the KL divergence loss between distributions for the active block.
    """
    p_dist = F.softmax(combined_logits[:, block_indices], dim=-1)
    q_dist = F.softmax(scorer_logits[:, block_indices], dim=-1)
    loss = F.kl_div(p_dist.log(), q_dist, reduction='batchmean')
    return loss

def sample_block(combined_logits, block_indices, temperature, mask_id, remasking):
    """
    Add noise to logits and select tokens; compute confidence scores.
    """
    combined_logits_noisy = add_gumbel_noise(combined_logits, temperature=temperature)
    x0 = torch.argmax(combined_logits_noisy[:, block_indices], dim=-1)
    
    if remasking == 'low_confidence':
        p = F.softmax(combined_logits[:, block_indices].to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(f"Remasking mode '{remasking}' is not implemented.")
    return x0, x0_p

def update_block(x, block_indices, x0, x0_p, num_transfer_tokens, step, mask_id):
    """
    For the current block, update only a subset of tokens based on confidence.
    """
    batch_size = x.shape[0]
    current_mask = (x[:, block_indices] == mask_id)
    x0 = torch.where(current_mask, x0, x[:, block_indices])
    confidence = torch.where(current_mask, x0_p, -float("inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
    for b in range(batch_size):
        k = num_transfer_tokens[b, step]
        _, topk_idx = torch.topk(confidence[b], k=k)
        transfer_index[b, topk_idx] = True
    with torch.no_grad():
        updated_block = x[:, block_indices].clone()
        updated_block[transfer_index] = x0[transfer_index]
        x[:, block_indices] = updated_block
    return x

def differentiable_generation(model,
                              tokenizer,
                              prompt_ids,
                              gen_length=64,
                              block_length=16,
                              steps=16,
                              score_model=None,
                              score_weight=1.0,
                              mask_id=126336,
                              lr=1e-2,
                              temperature=0.,
                              remasking='low_confidence',
                              progress_callback=None,
                              cfg_scale=0.):
    """
    Core differentiable generation function.
    """
    device = prompt_ids.device
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    total_len = prompt_len + gen_length

    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.clone()

    vocab_size = model.model.config.vocab_size
    delta = nn.Parameter(torch.zeros(batch_size, total_len, vocab_size, device=device))
    optimizer = torch.optim.SGD([delta], lr=lr)

    assert gen_length % block_length == 0, "gen_length must be a multiple of block_length."
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be a multiple of num_blocks."
    steps_per_block = steps // num_blocks

    total_steps = num_blocks * steps_per_block
    current_step = 0

    for block_i in range(num_blocks):
        block_start = prompt_len + block_i * block_length
        block_end = prompt_len + (block_i + 1) * block_length
        block_indices = torch.arange(block_start, block_end, device=device)
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            optimizer.zero_grad()
            combined_logits, scorer_logits = compute_combined_logits(
                model, x, delta, block_indices, score_model, score_weight, cfg_scale, mask_id
            )
            loss = gradient_update_step(combined_logits, scorer_logits, block_indices)
            loss.backward()
            optimizer.step()

            x0, x0_p = sample_block(combined_logits, block_indices, temperature, mask_id, remasking)
            x = update_block(x, block_indices, x0, x0_p, num_transfer_tokens, step, mask_id)

            current_step += 1
            if progress_callback is not None:
                progress_callback(current_step, total_steps, x, prompt_len, tokenizer, mask_id)
                time.sleep(0.01)
        optimizer.zero_grad()

    return x, delta

def classical_generation(model,
                         tokenizer,
                         prompt, 
                         steps=128, 
                         gen_length=128, 
                         block_length=128, 
                         temperature=0., 
                         cfg_scale=0., 
                         remasking='low_confidence', 
                         mask_id=126336, 
                         progress_callback=None):
    """
    Classical generation function with a progress bar.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_len = prompt.shape[1]
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "Steps must be divisible by the number of blocks."
    steps_per_block = steps // num_blocks

    total_steps = num_blocks * steps_per_block
    current_step = 0

    for num_block in range(num_blocks):
        block_slice = slice(prompt.shape[1] + num_block * block_length,
                            prompt.shape[1] + (num_block + 1) * block_length)
        block_mask_index = (x[:, block_slice] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                prompt_index = (x != mask_id)
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float("inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            current_step += 1
            if progress_callback is not None:
                progress_callback(current_step, total_steps, x, prompt_len, tokenizer, mask_id)
                time.sleep(0.01)
    return x
