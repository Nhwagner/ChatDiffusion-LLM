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
        hidden_states = self.transformer.wte(input_ids)
        hidden_states = self.transformer.emb_drop(hidden_states)
        for block in self.transformer.blocks:
            hidden_states = checkpoint(lambda x: block(x, **kwargs)[0], hidden_states)
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.transformer.ff_out(hidden_states)
        return SimpleNamespace(logits=logits)
