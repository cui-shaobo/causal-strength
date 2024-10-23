"""
@Project  : causalstrength
@File     : test_causal_strength.py
@Author   : Shaobo Cui
@Date     : 23.10.2024 20:18
"""
from causalstrength import evaluate

# For CESAR
s1 = "The rain caused the flooding."
s2 = "The streets were flooded."
cesar_score = evaluate(s1, s2, model_name='CESAR', model_path='cesar-model')
print(f'CESAR Causal strength: {cesar_score}')

# For CEQ
ceq_score = evaluate(s1, s2, model_name='CEQ', causes_path='data/causes.pkl', effects_path='data/effects.pkl')
print(f'CEQ Causal strength: {ceq_score}')
