"""
@Project  : causal-strength
@File     : test_causal_strength.py
@Author   : Shaobo Cui
@Date     : 23.10.2024 20:18
"""
from causalstrength import evaluate

# For CESAR
s1 = "Tom is very full now."
s2 = "He goes to McDonald for some food."
cesar_score = evaluate(s1, s2, model_name='CESAR', model_path='huggingfacesc/cesar-bert-large')
print(f'CESAR Causal strength: {cesar_score}')

# For CEQ
ceq_score = evaluate(s1, s2, model_name='CEQ', causes_path='data/causes.pkl', effects_path='data/effects.pkl')
print(f'CEQ Causal strength: {ceq_score}')
