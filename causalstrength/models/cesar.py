"""
@Project  : causalstrength
@File     : models.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 14:43
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig

class CESARConfig(AutoConfig):
    model_type = "cesar"

    def __init__(self, causal_attention=True, return_inner_scores=False, **kwargs):
        super().__init__(**kwargs)
        self.causal_attention = causal_attention
        self.return_inner_scores = return_inner_scores

class CESAR(PreTrainedModel):
    config_class = CESARConfig
    base_model_prefix = "cesar"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModelForCausalLM.from_pretrained('bert-large-uncased')

        if config.causal_attention:
            embedding_dim = self.encoder.config.hidden_size
            self.q = nn.Linear(embedding_dim, embedding_dim)
            self.k = nn.Linear(embedding_dim, embedding_dim)
        else:
            self.q = None
            self.k = None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        embeddings = outputs.last_hidden_state

        cs = []
        for embedding, attention, token_type in zip(embeddings, attention_mask, token_type_ids):
            tokens_mask = (attention == 1)
            first_sentence_emb = embedding[(tokens_mask) & (token_type == 0)]
            second_sentence_emb = embedding[(tokens_mask) & (token_type == 1)]

            first_norm = F.normalize(first_sentence_emb, p=2, dim=1)
            second_norm = F.normalize(second_sentence_emb, p=2, dim=1)

            score = torch.abs(first_norm @ second_norm.T)
            if self.q is not None and self.k is not None:
                q = self.q(first_sentence_emb)
                k = self.k(second_sentence_emb)
                attn = F.softmax(q @ k.T, dim=-1)
                cs.append(torch.sum(attn * score))
            else:
                cs.append(score.mean())

        cs = torch.stack(cs)
        return {'causal_strength': cs}
