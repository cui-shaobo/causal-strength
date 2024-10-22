"""
@Project  : causalstrength
@File     : models.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 14:43
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class CESAR(nn.Module):
    def __init__(self, model_name="YourUsername/cesar-model", causal_attention=True, return_inner_scores=False):
        super(CESAR, self).__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )

        self.return_inner_scores = return_inner_scores
        if causal_attention:
            embedding_dim = self.encoder.config.hidden_size
            self.q = nn.Linear(embedding_dim, embedding_dim)
            self.k = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embeddings = outputs.last_hidden_state

        cs = []
        for embedding, attention, token_type in zip(embeddings, attention_mask, token_type_ids):
            tokens_mask = (attention == 1)
            first_sentence_embedding_norm = F.normalize(embedding[torch.logical_and(tokens_mask, token_type == 0)], p=2, dim=1)
            second_sentence_embedding_norm = F.normalize(embedding[torch.logical_and(tokens_mask, token_type == 1)], p=2, dim=1)

            score = torch.abs(torch.matmul(first_sentence_embedding_norm.unsqueeze(0),
                                           second_sentence_embedding_norm.T.unsqueeze(0))).squeeze(0)
            if hasattr(self, 'q') and hasattr(self, 'k'):
                first_sentence_embedding = embedding[torch.logical_and(tokens_mask, token_type == 0)]
                second_sentence_embedding = embedding[torch.logical_and(tokens_mask, token_type == 1)]

                q = self.q(first_sentence_embedding)
                k = self.k(second_sentence_embedding)
                attn = q @ k.T
                attn = F.softmax(attn.flatten(), dim=0).view(attn.shape)

                if attn.shape != score.shape:
                    raise ValueError(f'Tensors must be of the same shape but got {attn.shape} and {score.shape}')

                cs.append(torch.sum(attn * score))
            else:
                cs.append(score.mean())

        cs = torch.stack(cs)
        if self.return_inner_scores and hasattr(self, 'q') and hasattr(self, 'k'):
            return cs, attn, score
        elif self.return_inner_scores:
            return cs, score
        else:
            return cs
