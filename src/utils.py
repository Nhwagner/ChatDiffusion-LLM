import torch
import torch.nn.functional as F
import time
import sys
import shutil
import math

def add_gumbel_noise(logits, temperature):
    """
    Add Gumbel noise to logits for sampling.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute number of tokens to update at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def live_progress_callback(current_step, total_steps, x, prompt_len, tokenizer, mask_id):
    """
    A progress callback that updates the same area in the terminal.
    """
    # On the first call, initialize state
    if not hasattr(live_progress_callback, "initialized"):
        live_progress_callback.initialized = True
        live_progress_callback.prev_lines_count = 0
        sys.stdout.write("Assistant:\n")
        sys.stdout.flush()

    gen_length = x.shape[1] - prompt_len
    num_generated = (x[:, prompt_len:] != mask_id).sum().item()
    progress_fraction = num_generated / gen_length
    bar_length = 20
    progress_bar = "[" + "#" * int(progress_fraction * bar_length) + "-" * (bar_length - int(progress_fraction * bar_length)) + "]"
    
    current_text = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    terminal_width = shutil.get_terminal_size().columns

    def count_wrapped_lines(text, width):
        lines = text.split('\n')
        count = 0
        for line in lines:
            count += max(1, math.ceil(len(line) / width))
        return count

    if current_step < total_steps:
        progress_line = f"Step {current_step}/{total_steps} {progress_bar} {progress_fraction*100:.1f}%"
        output_to_print = progress_line + "\n" + current_text
    else:
        output_to_print = current_text

    new_lines_count = count_wrapped_lines(output_to_print, terminal_width)
    for _ in range(live_progress_callback.prev_lines_count):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

    sys.stdout.write(output_to_print + "\n")
    sys.stdout.flush()
    live_progress_callback.prev_lines_count = new_lines_count

    if current_step >= total_steps:
        del live_progress_callback.prev_lines_count
        del live_progress_callback.initialized
