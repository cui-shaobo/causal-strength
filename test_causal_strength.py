"""
@Project  : causal-strength
@File     : test_causal_strength.py
@Author   : Shaobo Cui
@Date     : 23.10.2024 20:18
"""
# test_causal_strength.py

from causalstrength import evaluate

# Test CESAR Model
s1_cesar = "Tom is very full now."
s2_cesar = "He goes to McDonald for some food."

print("Testing CESAR model:")
cesar_score = evaluate(s1_cesar, s2_cesar, model_name='CESAR', model_path='huggingfacesc/cesar-bert-large')
print(f"CESAR Causal strength between \"{s1_cesar}\" and \"{s2_cesar}\": {cesar_score:.4f}")

# Test CEQ Model
s1_ceq = "Tom eats too much food."
s2_ceq = "He feels sick."

print("\nTesting CEQ model:")
ceq_score = evaluate(s1_ceq, s2_ceq, model_name='CEQ')
print(f"CEQ Causal strength between \"{s1_ceq}\" and \"{s2_ceq}\": {ceq_score:.4f}")

