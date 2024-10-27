"""
@Project  : causalstrength
@File     : test_heatmap.py
@Author   : Shaobo Cui
@Date     : 27.10.2024 16:40
"""

from causalstrength import evaluate

# Test CESAR Model
s1_cesar = "Fire starts quickly."
s2_cesar = "House burns to ashes."

print("Testing CESAR model:")
cesar_score = evaluate(s1_cesar, s2_cesar, model_name='CESAR', model_path='huggingfacesc/cesar-bert-large',
                       plot_heatmap_flag=True, heatmap_path=f'./figures/causal_heatmap.png')