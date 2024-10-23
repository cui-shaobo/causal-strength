"""
@Project  : causalstrength
@File     : plot_heatmap.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 15:34
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
# Set the backend to 'Agg' before importing pyplot
matplotlib.use('Agg')

import seaborn as sns
import os


def plot_heatmap(score_map, attn_map, cause_tokens, effect_tokens, causal_strength, save_path):
    """
    Generate and save heatmap plots for CESAR model evaluations.

    Parameters:
    - score_map (torch.Tensor): Embedding cosine similarities tensor with shape [cause_length, effect_length].
    - attn_map (torch.Tensor): Attention scores tensor with shape [cause_length, effect_length].
    - cause_tokens (list of str): Tokens from the cause statement.
    - effect_tokens (list of str): Tokens from the effect statement.
    - causal_strength (float): The causal strength score.
    - save_path (str): File path to save the heatmap plot.
    """
    # Convert tensors to numpy arrays and detach from computation graph
    score_map_np = score_map.detach().cpu().numpy()
    attn_map_np = attn_map.detach().cpu().numpy()
    result = score_map_np * attn_map_np

    # Apply thresholding to remove negligible values
    threshold = 1e-3
    score_map_np[score_map_np < threshold] = 0
    attn_map_np[attn_map_np < threshold] = 0
    result[result < threshold] = 0

    cause_tokens = ['CLS'] + cause_tokens + ['EOS']
    effect_tokens = effect_tokens + ['EOS']

    num_cause = len(cause_tokens)
    num_effect = len(effect_tokens)



    # Adjust figure size based on the number of tokens
    # Prevent excessively large or small figures
    figsize = max(18, num_effect * 0.5), max(6, num_cause * 0.5)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Heatmap for Embedding Cosine Similarities
    sns.heatmap(
        score_map_np,
        yticklabels=cause_tokens,
        xticklabels=effect_tokens,
        center=0.5,
        cmap='viridis',
        ax=axs[0],
        annot=False,  # Disable annotations for clarity
        fmt=".2f"
    )
    axs[0].set_title("Embedding Cosine Similarities", fontsize=14)
    axs[0].tick_params(axis='y', rotation=0)
    axs[0].tick_params(axis='x', rotation=90)

    # Heatmap for Attention Scores
    sns.heatmap(
        attn_map_np,
        yticklabels=cause_tokens,
        xticklabels=effect_tokens,
        center=0.5,
        cmap='viridis',
        ax=axs[1],
        annot=False,  # Disable annotations for clarity
        fmt=".2f"
    )
    axs[1].set_title("Attention Scores", fontsize=14)
    axs[1].tick_params(axis='y', rotation=0)
    axs[1].tick_params(axis='x', rotation=90)

    # Heatmap for Causal Strength (Product of Scores and Attention)
    # print(cause_tokens)
    # print(effect_tokens)

    print(cause_tokens)
    print(effect_tokens)
    sns.heatmap(
        result,
        yticklabels=cause_tokens,
        xticklabels=effect_tokens,
        center=0.5,
        cmap='viridis',
        ax=axs[2],
        annot=False,  # Disable annotations for clarity
        fmt=".2f"
    )
    axs[2].set_title(f"Causal Strength: {causal_strength:.2f}", fontsize=14)
    axs[2].tick_params(axis='y', rotation=0)
    axs[2].tick_params(axis='x', rotation=90)

    # Highlight Special Tokens

    # Adjust layout
    plt.tight_layout()

    # Ensure the figures directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

    print(f"Heatmap saved to {save_path}")