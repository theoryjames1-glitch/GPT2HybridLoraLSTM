# GPT2HybridLoraLSTM
```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class LSTMAdapter(nn.Module):
    """A lightweight LSTM adapter layer"""
    def __init__(self, hidden_size, lstm_hidden=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(lstm_hidden, hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return x + self.proj(lstm_out)  # residual add


class GPT2HybridLoraLSTM(nn.Module):
    def __init__(self, model_name="gpt2", insert_every=2, lstm_hidden=256):
        super().__init__()
        # Load GPT-2
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Apply LoRA to attention projections
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention projections
            lora_dropout=0.05,
            bias="none",
        )
        self.gpt2 = get_peft_model(base_model, lora_config)

        hidden_size = base_model.config.hidden_size

        # Add LSTM adapters every N transformer blocks
        self.lstm_adapters = nn.ModuleDict()
        for i, _ in enumerate(self.gpt2.base_model.transformer.h):
            if i % insert_every == 0:
                self.lstm_adapters[str(i)] = LSTMAdapter(
                    hidden_size, lstm_hidden=lstm_hidden
                )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embeddings
        inputs_embeds = (
            self.gpt2.base_model.transformer.wte(input_ids)
            + self.gpt2.base_model.transformer.wpe(
                torch.arange(input_ids.size(1), device=input_ids.device)
            )
        )
        hidden_states = inputs_embeds

        # Forward GPT-2 blocks + LoRA + LSTM adapters
        for i, block in enumerate(self.gpt2.base_model.transformer.h):
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
            if str(i) in self.lstm_adapters:
                hidden_states = self.lstm_adapters[str(i)](hidden_states)

        # Final norm + LM head
        hidden_states = self.gpt2.base_model.transformer.ln_f(hidden_states)
        logits = self.gpt2.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

```
