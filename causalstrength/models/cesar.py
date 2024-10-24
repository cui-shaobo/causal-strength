# modeling_cesar.py
import requests
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertConfig,
    PreTrainedModel,
    PretrainedConfig,
    logging as transformers_logging,
)

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from tqdm import tqdm
import os
import logging
# Configure logging to display info-level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
# import logging
#
# # Configure logging to display info-level messages (optional but recommended)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Optionally, set Transformers logging to INFO to see more details
# transformers_logging.set_verbosity_info()

def download_file_with_progress(url, local_path, desc):
    """
    Downloads a file from a URL to a local path with a progress bar.

    Parameters:
    - url (str): The URL to download the file from.
    - local_path (str): The local file path where the file will be saved.
    - desc (str): Description for the progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    with open(local_path, 'wb') as file, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=desc, leave=True
    ) as pbar:
        for data in response.iter_content(block_size):
            if data:  # Filter out keep-alive new chunks
                file.write(data)
                pbar.update(len(data))


class CESARConfig(PretrainedConfig):
    model_type = "cesar"

    def __init__(
        self,
        bert_model_name="bert-large-uncased",
        causal_attention=True,
        return_inner_scores=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.causal_attention = causal_attention
        self.return_inner_scores = return_inner_scores

        # Load the BertConfig
        self.bert_config = BertConfig.from_pretrained(
            bert_model_name, output_hidden_states=True
        )

class CESAR(PreTrainedModel):
    config_class = CESARConfig
    base_model_prefix = "cesar"

    def __init__(self, config):
        super().__init__(config)
        # Initialize the encoder with pre-trained weights
        self.encoder = BertModel.from_pretrained(config.bert_model_name, config=config.bert_config)
        self.return_inner_scores = config.return_inner_scores

        if config.causal_attention:
            embedding_dim = self.encoder.config.hidden_size
            self.q = nn.Linear(embedding_dim, embedding_dim)
            self.k = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        embeddings = outputs.last_hidden_state

        cs = []
        for embedding, attention, token_type in zip(
            embeddings, attention_mask, token_type_ids
        ):
            tokens_mask = attention.bool()
            first_sentence_mask = torch.logical_and(tokens_mask, token_type == 0)
            second_sentence_mask = torch.logical_and(tokens_mask, token_type == 1)

            first_sentence_embedding = embedding[first_sentence_mask]
            second_sentence_embedding = embedding[second_sentence_mask]

            first_norm = F.normalize(first_sentence_embedding, p=2, dim=1)
            second_norm = F.normalize(second_sentence_embedding, p=2, dim=1)

            score = torch.abs(
                torch.matmul(first_norm.unsqueeze(0), second_norm.T.unsqueeze(0))
            ).squeeze(0)

            if hasattr(self, "q") and hasattr(self, "k"):

                q = self.q(first_sentence_embedding)
                k = self.k(second_sentence_embedding)
                attn = q @ k.T
                attn = F.softmax(attn.flatten(), dim=0).view(attn.shape)

                if attn.shape != score.shape:
                    raise ValueError(
                        f"Tensors must be of the same shape but got {attn.shape} and {score.shape}"
                    )

                cs.append(torch.sum(attn * score))
            else:
                cs.append(score.mean())

        cs = torch.stack(cs)
        # print('*#' * 20)
        # print(self.return_inner_scores, hasattr(self, "q"), hasattr(self, "k"))
        if self.return_inner_scores and hasattr(self, "q") and hasattr(self, "k"):
            return cs, attn, score
        elif self.return_inner_scores:
            return cs, score
        else:
            return cs

    # @classmethod
    # def from_pretrained(cls, *args, **kwargs):
    #     # Load the custom config
    #     config = CESARConfig.from_pretrained(*args, **kwargs)
    #     # Initialize the model with pre-trained encoder
    #     model = super().from_pretrained(config.bert_model_name, config=config, *args, **kwargs)
    #     return model

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     """
    #     Overrides the from_pretrained method to handle 'progress' correctly.
    #
    #     Parameters:
    #     - pretrained_model_name_or_path (str): Model identifier from huggingface.co/models or path to local directory.
    #     - *model_args: Positional arguments.
    #     - **kwargs: Keyword arguments.
    #
    #     Returns:
    #     - CESAR: An instance of the CESAR model.
    #     """
    #     # Extract 'progress' from kwargs if present; default to True
    #     progress = kwargs.pop("progress", True)
    #
    #     # Initialize HfApi
    #     api = HfApi()
    #
    #     # Get the list of files in the model repository
    #     repo_info = api.model_info(pretrained_model_name_or_path)
    #     files = [f.rfilename for f in repo_info.siblings]
    #
    #     # Download each file with a progress bar
    #     local_dir = os.path.join(os.getcwd(), "temp_cesar_model")
    #     os.makedirs(local_dir, exist_ok=True)
    #
    #     print(f"Downloading model '{pretrained_model_name_or_path}' to '{local_dir}' with progress bar...")
    #     for filename in tqdm(files, desc="Downloading model files"):
    #         hf_hub_download(
    #             repo_id=pretrained_model_name_or_path,
    #             filename=filename,
    #             cache_dir=local_dir,
    #             local_dir=local_dir,
    #             resume_download=True,
    #         )
    #
    #     # Load the config from the downloaded files
    #     config_path = os.path.join(local_dir, "config.json")
    #     config = cls.config_class.from_pretrained(config_path, *model_args, **kwargs)
    #
    #     # Initialize the model without passing 'progress'
    #     model = cls(config)
    #
    #     # Load the encoder with the extracted 'progress' argument
    #     # Since we've already downloaded the files, progress isn't necessary here
    #     model.encoder = BertModel.from_pretrained(
    #         os.path.join(local_dir),
    #         config=config.bert_config,
    #         **kwargs  # Pass any other remaining kwargs to encoder
    #     )
    #
    #     # Clean up: Remove the temporary directory
    #     import shutil
    #     shutil.rmtree(local_dir)
    #
    #     return model