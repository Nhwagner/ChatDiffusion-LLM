import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import add_gumbel_noise, get_num_transfer_tokens, sentiment_guidance_loss

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)

def compute_combined_logits(model, x, delta, block_indices, cfg_scale, mask_id):
    """
    Compute model logits with optional classifier-free guidance and add delta.
    
    This function computes the output logits from the model while optionally applying 
    classifier-free guidance. When guidance is enabled (cfg_scale > 0), it creates a "null" input 
    version (where prompt tokens are masked) to compute the guided logits. It then adds an learned offset (delta) 
    
    Parameters:
      - model: Main language model that returns logits.
      - x: Input token tensor (includes prompt and masked tokens).
      - delta: Learnable offset tensor, added to adjust logits.
      - block_indices: Indices for the current block of tokens to update.
      - cfg_scale: Classifier-free guidance scale.
      - mask_id: Token id that indicates a masked token.
    
    Returns:
      - combined_logits: Logits adjusted with guidance and delta.
    """
    if cfg_scale > 0.:
        # Identify prompt positions where tokens are not masked.
        prompt_index = (x != mask_id)
        # Clone x and set prompt tokens to mask to create a null version.
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        # Concatenate the original and null inputs along the batch dimension.
        x_cat = torch.cat([x, un_x], dim=0)
        # Get logits for the concatenated input.
        logits_out = model(x_cat).logits
        # Split the logits into two halves: one for the original prompt and one for the null input.
        logits_main, un_logits = torch.chunk(logits_out, 2, dim=0)
        # Combine the logits using classifier-free guidance.
        main_logits = un_logits + (cfg_scale + 1) * (logits_main - un_logits)
    else:
        # If no guidance is used, simply get the logits from the model.
        main_logits = model(x).logits

    # Create a copy of the main logits to modify.
    combined_logits = main_logits.clone()
    # Add weighted score model logits and delta adjustments only at the active block positions.
    combined_logits[:, block_indices] += delta[:, block_indices]
    return combined_logits

def gradient_update_step(combined_logits, scorer_logits, block_indices):
    """
    Compute the KL divergence loss between distributions for the active block.
    
    This function computes the KL divergence between the probability distributions obtained 
    from the combined logits (with delta) and the scorer logits. The softmax function is applied 
    to both logits to get valid probability distributions over the token vocabulary.
    
    Parameters:
      - combined_logits: Logits from compute_combined_logits (includes delta adjustments).
      - scorer_logits: Logits from the score model or detached main logits.
      - block_indices: Indices for the current block to compute the loss over.
    
    Returns:
      - loss: The KL divergence loss computed over the active block.
    """
    # Convert combined logits to a probability distribution using softmax.
    p_dist = F.softmax(combined_logits[:, block_indices], dim=-1)
    # Convert scorer logits to a probability distribution.
    q_dist = F.softmax(scorer_logits[:, block_indices], dim=-1)
    # Calculate the KL divergence between the two distributions.
    loss = F.kl_div(p_dist.log(), q_dist, reduction='batchmean')
    return loss


def sample_block(combined_logits, block_indices, temperature, mask_id, remasking):
    """
    Add noise to logits and select tokens; compute confidence scores.
    
    This function applies Gumbel noise to the logits for stochastic sampling, then selects 
    token predictions for the active block using argmax. It also computes confidence scores 
    for the predictions based on the chosen remasking strategy.
    
    Parameters:
      - combined_logits: Logits after delta and score adjustments.
      - block_indices: Indices corresponding to the current block.
      - temperature: Temperature parameter to control noise intensity.
      - mask_id: Token id for masked positions.
      - remasking: Strategy for computing confidence ('low_confidence' or 'random').
    
    Returns:
      - x0: Predicted token ids for the block.
      - x0_p: Confidence scores corresponding to the predictions.
    """
    # Add Gumbel noise to the logits to enable stochastic sampling.
    combined_logits_noisy = add_gumbel_noise(combined_logits, temperature=temperature)
    # Predict token ids by taking the argmax over the noisy logits for the specified block.
    x0 = torch.argmax(combined_logits_noisy[:, block_indices], dim=-1)
    
    if remasking == 'low_confidence':
        # Compute softmax probabilities (confidence scores) from the original logits.
        p = F.softmax(combined_logits[:, block_indices].to(torch.float64), dim=-1)
        # Gather the probability corresponding to the selected token for each position.
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
    elif remasking == 'random':
        # Generate random confidence scores.
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        # If an unsupported remasking strategy is provided, raise an error.
        raise NotImplementedError(f"Remasking mode '{remasking}' is not implemented.")
    return x0, x0_p

def update_block(x, block_indices, x0, x0_p, num_transfer_tokens, step, mask_id):
    """
    Update only a subset of tokens in the current block based on confidence.
    
    This function selectively updates the masked tokens in the current block by comparing 
    confidence scores and updating only the top-k tokens (where k is defined per batch for each step).
    
    Parameters:
      - x: The current token sequence.
      - block_indices: Indices corresponding to the block being updated.
      - x0: New token predictions for the block.
      - x0_p: Confidence scores for the new predictions.
      - num_transfer_tokens: A tensor specifying the number of tokens to update per batch at each step.
      - step: The current update step within the block.
      - mask_id: The id used for masked tokens.
    
    Returns:
      - x: The updated token sequence with selected tokens replaced.
    """
    batch_size = x.shape[0]
    # Identify positions in the block that are still masked.
    current_mask = (x[:, block_indices] == mask_id)
    # For positions that are not masked, keep the current token (do not replace).
    x0 = torch.where(current_mask, x0, x[:, block_indices])
    # Set confidence scores to -inf for tokens that are not masked to prevent their update.
    confidence = torch.where(current_mask, x0_p, -float("inf"))
    # Initialize a boolean mask to track which tokens should be updated.
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
    # For each item in the batch, update the top-k tokens based on confidence.
    for b in range(batch_size):
        k = num_transfer_tokens[b, step]  # Number of tokens to update for this sample and step.
        _, topk_idx = torch.topk(confidence[b], k=k)
        transfer_index[b, topk_idx] = True
    # Update the token sequence x for the selected tokens.
    with torch.no_grad():
        updated_block = x[:, block_indices].clone()
        updated_block[transfer_index] = x0[transfer_index]
        x[:, block_indices] = updated_block

    
    return x

def differentiable_generation(model,
                              tokenizer,
                              prompt_ids,
                              score_prompt = None,
                              mode = "sentiment", #"sentiment", "LLM"
                              sentiment_label = 0,
                              gen_length=64,
                              block_length=16,
                              steps=16,
                              score_model=None,
                              score_tokenizer=None,
                              score_weight=1.0,
                              mask_id=126336,
                              lr=1e-2,
                              temperature=0.,
                              remasking='low_confidence',
                              progress_callback=None,
                              cfg_scale=0.):
    """
    Core differentiable generation function that uses gradient-based optimization of delta.
    
    This function generates tokens by iteratively optimizing a delta parameter added to the model's logits.
    It updates the token sequence block-by-block, applying classifier-free guidance (if enabled), and using a 
    confidence-based selection strategy to update only a subset of tokens per step.
    
    Parameters:
      - model: Main language model.
      - tokenizer: Tokenizer for converting tokens (used in progress callback).
      - prompt_ids: Initial prompt token ids.
      - gen_length: Number of tokens to generate.
      - score prompt: The tokens which will augment the output
      - mode: sentiment for sentiment signal, LLM for KL divergence
      - block_length: Number of tokens in each block.
      - steps: Total number of update steps (must be a multiple of the number of blocks).
      - score_model: Optional auxiliary model for scoring.
      - score_weight: Weight factor for score model logits.
      - mask_id: Id used for masked tokens.
      - lr: Learning rate for optimizing delta.
      - temperature: Temperature for adding Gumbel noise.
      - remasking: Strategy for remasking ('low_confidence' or 'random').
      - progress_callback: Optional callback to report progress.
      - cfg_scale: Classifier-free guidance scale.
    
    Returns:
      - x: Final generated token sequence.
      - delta: Optimized delta parameter tensor.
    """
    class ProjectionLayer(nn.Module):
      def __init__(self, input_dim, output_dim):
          super().__init__()
          self.proj = nn.Linear(input_dim, output_dim)

      def forward(self, x):
          return self.proj(x)
    
    device = prompt_ids.device

    if mode == "sentiment":
        projection = ProjectionLayer(4096, 768).to(device)
        projection = projection.to(model.dtype)
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    total_len = prompt_len + gen_length

    score_prompt_ids = tokenizer(score_prompt, return_tensors="pt").input_ids.to(device)

    # Initialize token sequence with all positions set to mask_id and then fill in the prompt.
    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.clone()

    # Create a delta tensor (initialized to zeros) for adjusting logits, with shape [batch, total_len, vocab_size].
    vocab_size = model.model.config.vocab_size
    delta = nn.Parameter(torch.zeros(batch_size, total_len, vocab_size, device=device))
    # Set up the optimizer (SGD) for delta.
    optimizer = torch.optim.SGD([delta], lr=lr)

    # Ensure generation length is divisible by block_length.
    assert gen_length % block_length == 0, "gen_length must be a multiple of block_length."
    num_blocks = gen_length // block_length
    # Ensure total steps is divisible by the number of blocks.
    assert steps % num_blocks == 0, "steps must be a multiple of num_blocks."
    steps_per_block = steps // num_blocks

    total_steps = num_blocks * steps_per_block
    current_step = 0

    # Process generation block by block.
    for block_i in range(num_blocks):
        
        # Define the token indices for the current block.
        block_start = prompt_len + block_i * block_length
        block_end = prompt_len + (block_i + 1) * block_length
        block_indices = torch.arange(block_start, block_end, device=device)
        # Determine which tokens in the block are still masked.
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        # Compute the number of tokens to transfer (update) per step in the block.
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # Iterate over the steps for the current block.
        for step in range(steps_per_block):

            # Compute logits with classifier-free guidance and add delta.
            combined_logits = compute_combined_logits(
                model, x, delta, block_indices, cfg_scale, mask_id
            )

            # Sample new token predictions and compute their confidence scores.
            x0, x0_p = sample_block(combined_logits, block_indices, temperature, mask_id, remasking)
            
            if mode == "sentiment":

                #Calculate sentiment score
                probs = torch.softmax(combined_logits, dim=-1)
                embedding_matrix = model.model.transformer.wte.weight
                weighted_embedding = torch.matmul(probs, embedding_matrix)
                projected_emb = projection(weighted_embedding)
                sentiment_outputs = score_model(inputs_embeds=projected_emb)

            if mode == "sentiment":
                #Apply sentiment loss, to update delta
                loss = sentiment_guidance_loss(sentiment_outputs, target_label=sentiment_label)
                
            else:
                # Compute the KL divergence loss between the modified logits and the score logits.
                score_input = torch.cat([score_prompt_ids, x], dim = 1)
                scorer_logits = model(score_input).logits[:,score_prompt_ids.shape[1]:]
                loss = gradient_update_step(combined_logits, scorer_logits, block_indices)

            # Backpropagate the loss to update delta.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update the token sequence x based on the new predictions and confidence.
            x = update_block(x, block_indices, x0, x0_p, num_transfer_tokens, step, mask_id)

            current_step += 1
            # Optionally report progress.
            if progress_callback is not None:
                progress_callback(current_step, total_steps, x, prompt_len, tokenizer, mask_id)
                time.sleep(0.01)
        # Clear gradients after finishing a block.
        optimizer.zero_grad()

    return x, delta

def classic_generation(model,
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
    Classic generation function with an iterative update of the token sequence.
    
    This function generates tokens using a more traditional iterative approach that directly updates 
    the token sequence based on model predictions. It divides the generation process into blocks and 
    updates tokens based on confidence scores (using either a softmax-based or random strategy).
    
    Parameters:
      - model: Main language model.
      - tokenizer: Tokenizer for converting tokens (used in progress callback).
      - prompt: Initial prompt token tensor.
      - steps: Total generation steps (must be divisible by the number of blocks).
      - gen_length: Number of tokens to generate.
      - block_length: Size of each block (number of tokens updated together).
      - temperature: Temperature for adding Gumbel noise.
      - cfg_scale: Classifier-free guidance scale.
      - remasking: Strategy for computing confidence ('low_confidence' or 'random').
      - mask_id: Token id representing masked tokens.
      - progress_callback: Optional callback for progress reporting.
    
    Returns:
      - x: Final token sequence after generation.
    """
    # Initialize token sequence with mask tokens and insert the prompt.
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_len = prompt.shape[1]
    num_blocks = gen_length // block_length
    # Ensure the total steps are divisible by the number of blocks.
    assert steps % num_blocks == 0, "Steps must be divisible by the number of blocks."
    steps_per_block = steps // num_blocks

    total_steps = num_blocks * steps_per_block
    current_step = 0

    # Iterate over each block.
    for num_block in range(num_blocks):
        # Define the slice for the current block.
        block_slice = slice(prompt.shape[1] + num_block * block_length,
                            prompt.shape[1] + (num_block + 1) * block_length)
        # Identify masked tokens in the current block.
        block_mask_index = (x[:, block_slice] == mask_id)
        # Compute the number of tokens to update for each step in the block.
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            # Identify all positions in x that are still masked.
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                # Apply classifier-free guidance:
                # Replace non-masked tokens with mask_id to form a null prompt.
                prompt_index = (x != mask_id)
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                # Concatenate original and null inputs.
                x_ = torch.cat([x, un_x], dim=0)
                # Compute logits and split into guided components.
    
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # Get logits directly from the model.
                logits = model(x).logits

            # Add Gumbel noise for stochasticity.
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # Predict tokens by taking the argmax.
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                # Compute softmax probabilities as confidence scores.
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                # Use random confidence scores.
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Ensure tokens outside the current block have very low confidence.
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -float("inf")
            # For already generated tokens, retain their original values.
            x0 = torch.where(mask_index, x0, x)
            # Set confidence to -infinity for positions that are not masked.
            confidence = torch.where(mask_index, x0_p, -float("inf"))
            # Create a mask to select which tokens to update.
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # For each sample, select the top tokens to update.
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            # Update the token sequence with new predictions at selected positions.
            x[transfer_index] = x0[transfer_index]
            current_step += 1
            # Optionally report progress.
            if progress_callback is not None:
                progress_callback(current_step, total_steps, x, prompt_len, tokenizer, mask_id)
                time.sleep(0.01)
    return x
