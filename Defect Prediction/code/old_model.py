import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from peft import PeftModel
from typing import Optional


class DecoderClassifier(nn.Module):
    """
    Frozen decoder-only encoder (e.g., CodeLlama/StarCoder) + linear head.
    Uses LAST NON-PAD token hidden state -> classifier.
    """
    def __init__(self, encoder, config, tokenizer, args, num_labels: int = 4):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.num_labels = num_labels

        # Ensure pad token is set for correct masking
        if getattr(self.config, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)
            self.config.pad_token_id = self.tokenizer.pad_token_id

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from config/encoder.")

        self.classifier = nn.Linear(hidden_size, num_labels)

        # 🔒 Freeze the backbone
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()  # disable dropout etc. in trunk

    @torch.no_grad()  # no grads through the frozen encoder
    def _encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Build attention mask if missing
        if attention_mask is None:
            pad_id = self.config.pad_token_id
            if pad_id is None:
                raise ValueError("pad_token_id must be set.")
            attention_mask = (input_ids != pad_id).to(input_ids.dtype)  # [B, T]

        # Frozen trunk forward (no grad)
        outputs = self._encode(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = getattr(outputs, "last_hidden_state", outputs[0])  # [B, T, H]

        # Pick LAST NON-PAD token per sequence
        lengths = attention_mask.long().sum(dim=1) - 1                   # [B]
        lengths = torch.clamp(lengths, min=0)
        b_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[b_idx, lengths]                              # [B, H]

        # Classifier (trainable)
        logits = self.classifier(pooled.to(self.classifier.weight.dtype)) # [B, C]

        if labels is not None:
            if isinstance(labels, list) or labels.dim() == 0:
                labels = torch.tensor(labels, device=logits.device, dtype=torch.long).view(-1)
            else:
                labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels, weight=weight)
            return loss, logits
        else:
            return logits