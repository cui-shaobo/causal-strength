"""
@Project  : causal-strength
@File     : evaluator.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 15:34
"""
import os

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.utils import is_remote_url


from . import CESAR
from .models import CEQ
from .utils.download_data import download_ceq_data
from .visualization.causal_heatmap import plot_heatmap

DEFAULT_DATA_DIR = os.path.expanduser('~/.causalstrength/data/')

CACHE_DIR = os.path.expanduser('~/.cache/huggingface/hub')

def model_is_cached(model_identifier):
    """
    Check if the Hugging Face model is already cached.
    Transforms the model identifier to match the 'models--' directory format.
    """
    # Transform the model identifier to match the cache directory's format
    model_dir_name = 'models--' + model_identifier.replace('/', '--')
    # List cached files and check if the model is present
    cached_files = os.listdir(CACHE_DIR)
    return any(model_dir_name in f for f in cached_files)


def evaluate(s1, s2, model_name='CESAR', model_path=None, device=None, plot_heatmap_flag=False, heatmap_path=None, **kwargs):
    """
    Evaluate causal strength between two statements using the specified model.
    Optionally, generate and save heatmap plots for CESAR model evaluations.

    Parameters:
    - s1 (str): The cause statement.
    - s2 (str): The effect statement.
    - model_name (str): Name of the model to use ('CESAR', 'CEQ', etc.).
    - model_path (str, optional): Hugging Face model identifier or local path for CESAR.
                                   For CEQ, this can be ignored or used for additional configurations.
    - device (torch.device, optional): Device to run the model on. Defaults to CUDA if available.
    - plot_heatmap_flag (bool, optional): Whether to generate and save heatmap plots (CESAR only).
    - heatmap_path (str, optional): File path to save the heatmap plots. If None and plotting is enabled, defaults to 'figures/causal_heatmap.pdf'.
    - kwargs: Additional arguments for model initialization.
              For CESAR:
                  - 'model_identifier' (str): Alias for 'model_path'.
              For CEQ:
                  - 'causes_path' (str): Path to the causes.pkl file.
                  - 'effects_path' (str): Path to the effects.pkl file.
                  - 'alpha' (float): Alpha parameter for CEQ calculation.
                  - 'lambda_' (float): Lambda parameter for CEQ calculation.

    Returns:
    - float or list of floats: The causal strength score(s).
    """
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'CESAR':
        model_identifier = model_path if model_path is not None else 'huggingfacesc/cesar-bert-large'

        # Check if the model is already cached locally
        if not model_is_cached(model_identifier):
            print(f"Downloading {model_identifier} model from Hugging Face hub...")
        # Load the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from '{model_identifier}'. Error: {e}")

        # Load the CESAR model using from_pretrained
        try:
            # print('*#'*40)
            # print('Start to load model from {}'.format(model_identifier))
            # print('*#'*40)
            model = CESAR.from_pretrained(model_identifier)
            # print('Finish the model loading process!')
            # print('*#'*40)
        except Exception as e:
            raise RuntimeError(f"Failed to load CESAR model from '{model_identifier}'. Error: {e}")

        # Move model to the specified device
        model.to(device)
        model.eval()

        # Tokenize the input statements without fixed max_length
        try:
            encoded_inputs = tokenizer(
                s1,
                s2,
                add_special_tokens=True,
                truncation=True,
                padding=True,  # Dynamic padding based on the longest sequence in the batch
                return_tensors="pt"
            )
        except Exception as e:
            raise ValueError(f"Tokenization failed. Error: {e}")

        # Move tensors to the specified device
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        token_type_ids = encoded_inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Perform inference
        try:
            outputs = model(input_ids, attention_mask, token_type_ids)
        except Exception as e:
            raise RuntimeError(f"Inference failed. Error: {e}")

        # Handle different output formats based on 'return_inner_scores'
        if isinstance(outputs, tuple):
            # If 'return_inner_scores' is True, extract the primary causal strength (cs) and additional outputs
            score, attn_map, score_map = outputs
        else:
            # If 'return_inner_scores' is False, only the causal strength score is returned
            score = outputs
            attn_map = None
            score_map = None

        # Assuming batch_size=1, extract the scalar value
        if isinstance(score, torch.Tensor):
            if score.dim() == 1 and score.size(0) == 1:
                causal_strength = score.squeeze().item()
            else:
                # If multiple scores are returned (e.g., batch processing), convert to list
                causal_strength = score.cpu().tolist()
        else:
            # In case score is not a tensor
            causal_strength = score

        # If heatmap plotting is requested and model returned attention and score maps
        if plot_heatmap_flag and attn_map is not None and score_map is not None:
            # Decode tokens for labeling
            token_ids = encoded_inputs['input_ids'][0]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Use token_type_ids to distinguish between cause and effect tokens
            if token_type_ids is not None:
                token_type_ids_np = token_type_ids.cpu().numpy()[0]  # Shape: [seq_length]
                cause_indices = token_type_ids_np == 0  # Typically, token_type_id=0 for the first sentence
                effect_indices = token_type_ids_np == 1  # token_type_id=1 for the second sentence
            else:
                # If token_type_ids are not available, assume first half tokens are cause and second half are effect
                total_length = len(tokens)
                half = total_length // 2
                cause_indices = [True] * half + [False] * (total_length - half)
                effect_indices = [False] * half + [True] * (total_length - half)

            # Extract cause and effect tokens, excluding special tokens
            cause_tokens = [token.replace('##', '') for token, valid in zip(tokens, cause_indices) if valid and token not in ('[CLS]', '[SEP]')]
            effect_tokens = [token.replace('##', '') for token, valid in zip(tokens, effect_indices) if valid and token not in ('[CLS]', '[SEP]')]

            # Print shapes for debugging
            print(f"Original score_map shape: {score_map.shape}")
            print(f"Original attn_map shape: {attn_map.shape}")
            print(f"Number of cause tokens: {len(cause_tokens)}")
            print(f"Number of effect tokens: {len(effect_tokens)}")

            # Determine if score_map is 3D or 2D
            if score_map.dim() == 3:
                # Assume [batch, cause_seq, effect_seq]
                score_map = score_map[0, cause_indices, :][:, effect_indices]
                attn_map = attn_map[0, cause_indices, :][:, effect_indices]
            elif score_map.dim() == 2:
                # Assume [cause_seq, effect_seq]
                # print(cause_indices, effect_indices)
                # print(score_map.size())
                # score_map = score_map[cause_indices, :][:, effect_indices]
                # attn_map = attn_map[cause_indices, :][:, effect_indices]
                pass
            else:
                raise ValueError(f"Unexpected score_map dimensions: {score_map.dim()}")

            # Print sliced shapes for verification
            print(f"Sliced score_map shape: {score_map.shape}")
            print(f"Sliced attn_map shape: {attn_map.shape}")

            # Check if the sliced score_map and attn_map match the number of tokens
            if score_map.shape[0] != len(cause_tokens) or score_map.shape[1] != len(effect_tokens):
                print("Warning: The sliced score_map dimensions do not match the number of tokens.")

            # Set default heatmap path if not provided
            if heatmap_path is None:
                # Create a unique filename based on the input sentences
                safe_s1 = ''.join(c for c in s1 if c.isalnum() or c in (' ', '_')).rstrip()
                safe_s2 = ''.join(c for c in s2 if c.isalnum() or c in (' ', '_')).rstrip()
                heatmap_path = f'figures/causal_heatmap_{safe_s1.replace(" ", "_")}_to_{safe_s2.replace(" ", "_")}.pdf'

            # Generate heatmap
            # print('*#' * 20)
            plot_heatmap(score_map, attn_map, cause_tokens, effect_tokens, causal_strength, heatmap_path)

        return causal_strength

    elif model_name == 'CEQ':
        causes_path = kwargs.get('causes_path', os.path.join(DEFAULT_DATA_DIR, 'causes.pkl'))
        effects_path = kwargs.get('effects_path', os.path.join(DEFAULT_DATA_DIR, 'effects.pkl'))
        alpha = kwargs.get('alpha', 0.66)
        lambda_ = kwargs.get('lambda_', 1.0)

        # Check if causes and effects files exist, download if missing
        if not os.path.exists(causes_path) or not os.path.exists(effects_path):
            print(f"Downloading CEQ data to {DEFAULT_DATA_DIR}...")
            os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
            download_ceq_data(data_dir=DEFAULT_DATA_DIR)

        if not os.path.exists(causes_path) or not os.path.exists(effects_path):
            raise FileNotFoundError(f"Data files not found at {causes_path} or {effects_path}.")

        # Initialize the CEQ model
        try:
            model = CEQ(causes_path=causes_path, effects_path=effects_path, alpha=alpha, lambda_=lambda_)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CEQ model. Error: {e}")

        # Compute the causal strength score
        try:
            score = model.compute_score(s1, s2)
        except Exception as e:
            raise RuntimeError(f"Failed to compute causal strength using CEQ. Error: {e}")

        return score

    else:
        raise ValueError(f'Unknown model name: {model_name}')
