<small>中文 | [EN](README.md) </small>
# causal-strength  ![因果强度](https://img.shields.io/badge/causal--strength-%E2%9A%96%EF%B8%8F%20%E5%9B%A0%E6%9E%9C%E5%BC%BA%E5%BA%A6%E8%AF%84%E4%BC%B0-blue)  衡量因果关系的强度

<a href="https://aclanthology.org/2024.findings-acl.384/">
    <img src="https://img.shields.io/badge/2024.findings-acl.384-blue.svg?style=flat-square" alt="ACL Anthology" />
</a>
<a href="https://pypi.org/project/causal-strength/">
    <img src="https://img.shields.io/pypi/v/causal-strength?style=flat-square" alt="PyPI version" />
</a>



**causal-strength** 是一个用于评估陈述之间因果强度的 Python 包，使用如 CESAR（带注意力重加权的因果嵌入相似性）等多种指标。该包利用 [Hugging Face Transformers](https://huggingface.co/) 上的预训练模型，提供高效和可扩展的计算。

## 目录

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [📜 引用 *](#-引用-)
- [🌟 特性 *](#-特性-)
- [🚀 安装 *](#-安装-)
  - [前提条件](#前提条件)
  - [步骤](#步骤)
- [🛠️ 使用  *](#-使用-)
  - [快速开始](#快速开始)
  - [评估因果强度](#评估因果强度)
  - [生成因果热力图](#生成因果热力图)
- [参考文献](#参考文献)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 📜 引用 ![引用](https://img.shields.io/badge/引用-必需-green) 

如果你觉得这个包有帮助，请给我们的仓库 [causal-strength](https://github.com/cui-shaobo/causal-strength) 和相关仓库 [defeasibility-in-causality](https://github.com/cui-shaobo/defeasibility-in-causality) 点星。如果用于学术目的，请引用我们的论文：

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

## 🛠️ 使用方法  ![Usage](https://img.shields.io/badge/Usage-Instructions-green)

### 评估因果强度

`evaluate` 函数用于计算两个陈述之间的因果强度。

1. 使用 CESAR 模型:

    ```python
    from causalstrength import evaluate
    
    # Define your statements
    cause = "Fire starts."
    effect = "House burns."
    
    # Evaluate causal strength
    score = evaluate(cause, effect, model_name='CESAR', model_path='YourUsername/cesar-model')
    print(f'Causal strength between "{cause}" and "{effect}" is {score:.2f}')
    ```
    
2. 使用 CEQ 模型:

   ```python
    from causalstrength import evaluate

    # Test CEQ Model
    s1_ceq = "Tom eats too much food."
    s2_ceq = "He feels sick."
    
    print("\nTesting CEQ model:")
    ceq_score = evaluate(s1_ceq, s2_ceq, model_name='CEQ')
    print(f"CEQ Causal strength between \"{s1_ceq}\" and \"{s2_ceq}\": {ceq_score:.4f}")
    ```

**参数说明：**

- `s1` (str): 原因陈述。
- `s2` (str): 结果陈述。
- `model_name` (str): 使用的模型名称（如 `'CESAR'`, `'CEQ'` 等）。
- `model_path` (str): Hugging Face 模型标识符或模型的本地路径。

### 生成因果热力图

使用热力图可视化注意力和词汇间的相似度评分。

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
