## Setup for CEQ Model

The CEQ model requires two large data files: `causes.pkl` and `effects.pkl`, each approximately 1GB in size. These files are not included in the package and must be downloaded separately.

### **Download Data Files**

You can download the data files using the provided function:

```python
from causalstrength.utils.download_data import download_ceq_data

download_ceq_data(data_dir='data')  # Specify the directory where you want to save the files
```

Alternatively, you can use the script:
```bash
python scripts/download_ceq_data.py
```




## Usage
When evaluating causal strength using the CEQ model, specify the paths to the downloaded data files:
```python
from causalstrength import evaluate

s1 = "Smoking causes cancer."
s2 = "Cancer develops."

score = evaluate(
    s1,
    s2,
    model_name='CEQ',
    causes_path='data/causes.pkl',
    effects_path='data/effects.pkl'
)
print(f'Causal strength: {score:.4f}')
```
