"""
@Project  : causalstrength
@File     : evaluator.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 15:34
"""


import torch
from transformers import AutoTokenizer
from .models import CESAR  # Include CEQ when implemented

def evaluate(s1, s2, model_name='CESAR', model_path=None, device=None):
    """
    Evaluate causal strength between two statements using the specified model.

    Parameters:
    - s1 (str): The cause statement.
    - s2 (str): The effect statement.
    - model_name (str): Name of the model to use ('CESAR', 'CEQ', etc.).
    - model_path (str, optional): Hugging Face model identifier or local path.
    - device (torch.device, optional): Device to run the model on.

    Returns:
    - float: The causal strength score.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'CESAR':
        if model_path is None:
            model_identifier = 'YourUsername/cesar-model'  # Replace with your model identifier
        else:
            model_identifier = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        model = CESAR(model_name=model_identifier, causal_attention=True)
    elif model_name == 'CEQ':
        # Initialize CEQ model and tokenizer
        pass
    else:
        raise ValueError(f'Unknown model name: {model_name}')

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

    with torch.no_grad():
        score = model(input_ids, attention_mask, token_type_ids)
    return score.cpu().item()
