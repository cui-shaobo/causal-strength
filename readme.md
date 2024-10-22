# causalstrength

**causalstrength** is a Python package for evaluating the causal strength between statements using various metrics such as CESAR (Causal Embedding Similarity with Attention Reweighting). This package leverages pre-trained models available on [Hugging Face Transformers](https://huggingface.co/) for efficient and scalable computations.

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

## Features

- **Causal Strength Evaluation**: Compute the causal strength between two statements using models like CESAR.
- **Visualization Tools**: Generate heatmaps to visualize attention and similarity scores between tokens.
- **Extensibility**: Easily add new metrics and models for evaluation.
- **Hugging Face Integration**: Load models directly from the Hugging Face Model Hub.

## Installation

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) (for GPU support, ensure CUDA is properly configured)
- [Git](https://git-scm.com/)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/causalstrength.git
   cd causalstrength
   ```

2. **Install the Package**


## Quick Start
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

## Usage

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

## Acknowledgments
+ HuggingFace Transformers - For providing the model hub and transformer implementations
+ PyTorch - For providing the deep learning framework
