import torch
from transformers import AutoTokenizer

from causalstrength import CESAR
from causalstrength.models import CESARConfig

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('huggingfacesc/cesar-bert-large')

# Load the model
config = CESARConfig.from_pretrained('huggingfacesc/cesar-bert-large')
model = CESAR.from_pretrained('huggingfacesc/cesar-bert-large', config=config)

# Prepare inputs
s1 = "Fire starts."
s2 = "House burns."
encoded_inputs = tokenizer(
    s1,
    s2,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=128,
    add_special_tokens=True
)
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
token_type_ids = encoded_inputs['token_type_ids']

# Evaluate causal strength
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask, token_type_ids)
    causal_strength = outputs['causal_strength']

print(f'Causal strength between \"{s1}\" and \"{s2}\" is {causal_strength.item()}')
