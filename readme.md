# causal-strength  ![Causal Strength](https://img.shields.io/badge/causal--strength-%E2%9A%96%EF%B8%8F%20measurement%20of%20causality-blue)  Measure the Strength Between Cause and Effect

<a href="https://aclanthology.org/2024.findings-acl.384/">
    <img src="https://img.shields.io/badge/2024.findings-acl.384-blue.svg?style=flat-square" alt="ACL Anthology" />
</a>
<a href="https://pypi.org/project/causal-strength/">
    <img src="https://img.shields.io/pypi/v/causal-strength?style=flat-square" alt="PyPI version" />
</a>



**causal-strength** is a Python package for evaluating the causal strength between statements using various metrics such as CESAR (Causal Embedding Similarity with Attention Reweighting). This package leverages pre-trained models available on [Hugging Face Transformers](https://huggingface.co/) for efficient and scalable computations.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Evaluating Causal Strength](#evaluating-causal-strength)
  - [Generating Causal Heatmaps](#generating-causal-heatmaps)
- [Adding New Metrics](#adding-new-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 📜 Citation ![Citation](https://img.shields.io/badge/Citation-Required-green) 

If you find this package helpful, please star this repository [causal-strength](https://github.com/cui-shaobo/causal-strength) and the related repository: [defeasibility-in-causality](https://github.com/cui-shaobo/defeasibility-in-causality). For academic purposes, please cite our paper:

```bibtex
@inproceedings{cui-etal-2024-exploring,
    title = "Exploring Defeasibility in Causal Reasoning",
    author = "Cui, Shaobo  and
      Milikic, Lazar  and
      Feng, Yiyang  and
      Ismayilzada, Mete  and
      Paul, Debjit  and
      Bosselut, Antoine  and
      Faltings, Boi",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.384",
    doi = "10.18653/v1/2024.findings-acl.384",
    pages = "6433--6452",
}
```


## 🌟 Features ![Key Features](https://img.shields.io/badge/Key_Features-Highlights-orange) 

- **Causal Strength Evaluation**: Compute the causal strength between two statements using models like CESAR.
- **Visualization Tools**: Generate heatmaps to visualize attention and similarity scores between tokens.
- **Extensibility**: Easily add new metrics and models for evaluation.
- **Hugging Face Integration**: Load models directly from the Hugging Face Model Hub.

## 🚀 Installation ![Installation](https://img.shields.io/badge/Installation-Guide-blue)

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) (for GPU support, ensure CUDA is properly configured)
- [Git](https://git-scm.com/)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/causalstrength.git
   cd causal-strength
   ```

2. **Install the Package**




## 🛠️ Usage  ![Usage](https://img.shields.io/badge/Usage-Instructions-green)

### Quick Start
Here's a quick example to evaluate the causal strength between two statements:

```python
from causalstrength import evaluate

# Define your statements
cause = "Fire starts."
effect = "House burns."

# Evaluate causal strength
score = evaluate(cause, effect, model_name='CESAR', model_path='YourUsername/cesar-model')
print(f'Causal strength between "{cause}" and "{effect}" is {score:.2f}')
```

### Evaluating Causal Strength

The `evaluate` function computes the causal strength between two statements.

```python
from causalstrength import evaluate

# Statements to evaluate
s1 = "Rain falls."
s2 = "Ground gets wet."

# Evaluate using CESAR model from Hugging Face
score = evaluate(s1, s2, model_name='CESAR', model_path='YourUsername/cesar-model')
print(f'Causal strength: {score:.2f}')
```

**Parameters:**

- `s1` (str): The cause statement.
- `s2` (str): The effect statement.
- `model_name` (str): The name of the model to use (`'CESAR'`, `'CEQ'`, etc.).
- `model_path` (str): Hugging Face model identifier or local path to the model.

### Generating Causal Heatmaps

Visualize the attention and similarity scores between tokens using heatmaps.

```python
from causalstrength.visualization.causal_heatmap import generate_causal_heatmap

# Statements to visualize
s1 = "Fire starts."
s2 = "House burns."

# Generate heatmap
generate_causal_heatmap(
    s1,
    s2,
    model_name='YourUsername/cesar-model',
    save_path='causal_heatmap.pdf'
)
```

[//]: # (## Acknowledgments)

[//]: # (+ HuggingFace Transformers - For providing the model hub and transformer implementations)

[//]: # (+ PyTorch - For providing the deep learning framework)

## References
