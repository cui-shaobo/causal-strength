"""
@Project  : causal-strength
@File     : setup.py
@Author   : Shaobo Cui
@Date     : 22.10.2024 15:49
"""

from setuptools import setup, find_packages, Command

# Custom command to download and verify the CEQ data files
class DownloadDataCommand(Command):
    """A custom command to download and verify the CEQ data files."""
    description = 'Download and verify causes and effects data'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from causalstrength.utils.download_data import download_ceq_data
        download_ceq_data()

readme = open('README.md').read()


setup(
    name='causal-strength',
    version='0.1.0',
    description='A package for evaluating causal strength intensity between cause and effect.',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Shaobo Cui',
    author_email='shaobo.cui@epfl.ch',
    url='https://github.com/cui-shaobo/causal-strength',  # Update with your repository URL
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'transformers>=4.0.0',
        'matplotlib',
        'seaborn',
        'nltk',
        'gdown',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    cmdclass={
        'download_data': DownloadDataCommand,  # Custom command for downloading CEQ data
    },
    entry_points={
        'console_scripts': [
            'download_ceq_data=causalstrength.utils.download_data:download_ceq_data',  # CLI command to download data
        ],
    },
)