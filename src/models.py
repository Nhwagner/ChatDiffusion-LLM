import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint

class CheckpointedLLaDAModelLM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transformer = self.model.model.transformer

    def forward(self, input_ids, **kwargs):
        # Compute token embeddings using the word token embedding layer.
        hidden_states = self.transformer.wte(input_ids)
        # Apply dropout to the embeddings to help regularize the model.
        hidden_states = self.transformer.emb_drop(hidden_states)
        # Process the embeddings through each transformer block using gradient checkpointing.
        # Checkpointing allows saving memory by recomputing activations during backpropagation.
        for block in self.transformer.blocks:
            hidden_states = checkpoint(lambda x: block(x, **kwargs)[0], hidden_states, use_reentrant=True)
        # Apply layer normalization to the final hidden states.
        hidden_states = self.transformer.ln_f(hidden_states)
        # Compute output logits using the final feed-forward layer.
        logits = self.transformer.ff_out(hidden_states)
        # Return the logits wrapped in a SimpleNamespace for convenient attribute access.
        return SimpleNamespace(logits=logits)