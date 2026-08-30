# Always standardize your ML features?

Preprocessing, model choice, and optimization aren’t independent decisions.

They’re one system.

Standardization changes the geometry of the learning problem, so its effect depends on the estimator, solver/optimizer, and objective.

“Always standardize” is too simple.

“Never standardize” is too naive.

Standardize for what model, what optimizer, and what objective?

That is the question.

## Data and Setting

 3-fold CV, 50,000 MNIST images (shuffled from the 60k training set with a fixed seed), mean ± sd over seeds 0/1/2.

 Fixed budget for each model: logistic regression max_iter=2000, SGDClassifier 20, MLPs 40 epochs. Raw and standardized always get identical budgets.


## Results


1 SGDClassifier: 0.9021 → 0.9100 (+0.79 pp)

2️ Logistic regression (lbfgs): 0.9165 → 0.9037 (−1.28 pp)

3️ MLP-128 (Adam): 0.9737 → 0.9688 (−0.49 pp)

4️ MLP-128 (SGD): 0.9256 → 0.9483 (+2.27 pp)

5️ MLP-128 (Adam), 100 of 784 columns ×1,000: 0.9170 → 0.9690 (+5.20 pp)

6️ Logistic regression (lbfgs), 100 of 784 columns ×1,000: 0.8869 → 0.9035 (+1.67 pp)




## Reproduce

```
pip install -r requirements.txt
marimo edit mnist_scaling_study.py
```


## Files

| file | what it is |
|---|---|
| `mnist_scaling_study.py` | marimo notebook — all experiments, reproducible |
| `make_figure.py` | matplotlib script |
| `standardization_summary.png` | the chart summary |
| `README.md` | summary of the test |
| `requirements.txt` | pinned versions |
| `LICENSE.txt` | license file |


MIT licensed.
