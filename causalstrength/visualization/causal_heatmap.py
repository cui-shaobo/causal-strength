"""
@Project  : causalstrength
@File     : causal_heatmap.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 15:47
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from ..models.cesar import CESAR

def generate_causal_heatmap(s1, s2, model_name='YourUsername/cesar-model', save_path='causal_heatmap.pdf'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CESAR(model_name=model_name, causal_attention=True, return_inner_scores=True)
    model.to(device)
    model.eval()

    encoded_inputs = tokenizer(
        s1,
        s2,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    token_type_ids = encoded_inputs.get('token_type_ids', None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # Run the model and obtain scores
    score, attn_map, score_map = model(input_ids, attention_mask, token_type_ids)
    result = score_map * attn_map

    # Process attention and score maps for visualization
    attn_map[attn_map < 1e-3] = 0
    score_map[score_map < 1e-3] = 0
    result[result < 1e-3] = 0

    # Get tokens for labels
    tokens_s1 = tokenizer.tokenize(s1)
    tokens_s2 = tokenizer.tokenize(s2)

    # Plot heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(score_map.cpu().numpy(), yticklabels=tokens_s1, xticklabels=tokens_s2, ax=axs[0])
    axs[0].set_title('Embedding Cosine Similarities')
    sns.heatmap(attn_map.cpu().numpy(), yticklabels=tokens_s1, xticklabels=tokens_s2, ax=axs[1])
    axs[1].set_title('Attention Scores')
    sns.heatmap(result.cpu().numpy(), yticklabels=tokens_s1, xticklabels=tokens_s2, ax=axs[2])
    axs[2].set_title(f'Causal Strength: {score.item():.2f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
