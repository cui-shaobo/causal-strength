"""
@Project  : causal-strength
@File     : setup.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 15:49
"""

from setuptools import setup, find_packages

setup(
    name='causal-strength',
    version='0.1.0',
    description='A package for evaluating causal strength intensity between cause and effect.',
    author='Shaobo Cui',
    author_email='shaobo.cui@epfl.ch',
    url='https://github.com/cui-shaobo/causal-strength',  # Update with your repository URL
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'transformers>=4.0.0',
        'matplotlib',
        'seaborn',
        'nltk'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)