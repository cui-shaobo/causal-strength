import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nltk.stem as ns
import os
import pickle

import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class CEQ(nn.Module):
    """
    CEQ model for evaluating causal strength using statistical co-occurrence.

    Parameters:
    - causes_path (str): Path to the causes.pkl file.
    - effects_path (str): Path to the effects.pkl file.
    - alpha (float): Alpha parameter for CEQ calculation.
    - lambda_ (float): Lambda parameter for CEQ calculation.
    """

    def __init__(self, causes_path=None, effects_path=None, alpha=0.66, lambda_=1):
        super(CEQ, self).__init__()
        self.alpha = alpha
        self.lambda_ = lambda_

        # Set default paths if not provided
        if causes_path is None:
            causes_path = os.path.join('data', 'causes.pkl')
        if effects_path is None:
            effects_path = os.path.join('data', 'effects.pkl')

        # Check if data files exist
        if not os.path.exists(causes_path):
            raise FileNotFoundError(f"Causes data file not found at {causes_path}. "
                                    "Please download it using the provided script or function.")
        if not os.path.exists(effects_path):
            raise FileNotFoundError(f"Effects data file not found at {effects_path}. "
                                    "Please download it using the provided script or function.")

        # Load causes and effects dictionaries
        with open(causes_path, 'rb') as f:
            self.causes = pickle.load(f)

        with open(effects_path, 'rb') as f:
            self.effects = pickle.load(f)

        self.M = 62675002  # Total number of co-occurrences (adjust as necessary)
        self.lemmatizer = ns.WordNetLemmatizer()

    def tokenize(self, sent):
        sent = sent.lower()
        sent = sent.strip('.')
        sent = sent.replace("'s", '')
        words = sent.split(' ')
        lemmatized_words = []
        for word in words:
            word_n = self.lemmatizer.lemmatize(word, pos='n')
            word_v = self.lemmatizer.lemmatize(word, pos='v')
            lemmatized_words.append(word_v if word_n != word_v else word_n)
        return lemmatized_words

    def cs_word(self, w_cause, w_effect):
        try:
            p_w_cause = float(sum(self.causes[w_cause].values())) / self.M
        except KeyError:
            p_w_cause = 0

        try:
            p_w_effect = float(sum(self.effects[w_effect].values())) / self.M
        except KeyError:
            p_w_effect = 0

        try:
            p_join = float(self.causes[w_cause][w_effect]) / self.M
        except KeyError:
            p_join = 0

        if p_join != 0 and p_w_cause != 0 and p_w_effect != 0:
            cs_nes = p_join / (p_w_cause ** self.alpha) / p_w_effect
            cs_surf = p_join / p_w_cause / (p_w_effect ** self.alpha)
            cs = (cs_nes ** self.lambda_) * (cs_surf ** (1 - self.lambda_))
        else:
            cs = float(2) / len(self.causes)
        return cs

    def compute_score(self, s1, s2):
        """
        Compute the causal strength between two sentences.

        Parameters:
        - s1 (str): Cause sentence.
        - s2 (str): Effect sentence.

        Returns:
        - float: Causal strength score.
        """
        s_cause = self.tokenize(s1)
        s_effect = self.tokenize(s2)
        cs = 0
        for w_cause in s_cause:
            for w_effect in s_effect:
                cs += self.cs_word(w_cause, w_effect)
        cs /= (len(s_cause) + len(s_effect))
        return cs
