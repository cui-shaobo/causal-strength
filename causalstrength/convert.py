import torch
from transformers import AutoTokenizer

from causalstrength import CESAR
from causalstrength.models import CESARConfig

# Initialize the config and model
config = CESARConfig(causal_attention=True)
model = CESAR(config)

# Load your pre-trained weights
model.load_state_dict(torch.load('models/cesar_model.pt'))

# Save the model and tokenizer
model.save_pretrained('cesar-model')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
tokenizer.save_pretrained('cesar-model')
