# Representation Engineering in the Legal Domain: Linear and Non-Linear Probing of Jurema-7B

This repository contains the supplementary material for the paper *"Representation Engineering in the Legal Domain: A Comparative Study of Linear and Non-Linear Probing for Initial Petition Classification"*, submitted to ENIAC 2026.

The study compares linear and non-linear probing applied to the hidden states of the [Jurema-7B](https://huggingface.co/Jurema-br/Jurema-7B) model to classify Repetitive Demand Resolution Incident (IRDR) petitions from a Brazilian court into seven categories.

## Repository Structure

```
.
├── notebooks/
│   ├── Linear_Probing.ipynb        # Training and hyperparameter search for linear probing
│   └── Non_Linear_Probing.ipynb    # Training and hyperparameter search for non-linear (MLP) probing
└── results/
    ├── hyperparameters_linear_probing.csv
    └── hyperparameters_non_linear_probing.csv
```

## Hyperparameter Optimization

Hyperparameter optimization was conducted with the Tree-structured Parzen Estimator (TPE) via the Optuna framework, using 2-fold cross-validation per trial to select hyperparameters, followed by a final evaluation using stratified 5-fold cross-validation.

- **Linear probing:** 50 trials per scenario, searching learning rate and L2 penalty (weight decay).
- **Non-linear probing:** 100 trials per scenario, searching hidden layer size, learning rate, dropout rate, and weight decay.

The full set of selected hyperparameters for each of the 8 evaluated scenarios (2 summarization strategies × 2 layers × RAG/no-RAG) is available in [`results/hyperparameters_linear_probing.csv`](results/hyperparameters_linear_probing.csv) and [`results/hyperparameters_non_linear_probing.csv`](results/hyperparameters_non_linear_probing.csv).

### Non-Linear Probing — Selected Hyperparameters

| Configuration | ma-F1 | Std | Hidden Size | LR | Dropout | Weight Decay | Batch Size |
|---|---|---|---|---|---|---|---|
| Head+Tail \| Final (28th) \| No RAG | 0.617 | 0.021 | 128 | 1.4e-4 | 0.30 | 1.2e-4 | 64 |
| Head+Tail \| Final (28th) \| RAG | 0.600 | 0.010 | 128 | 4.2e-3 | 0.50 | 3.8e-4 | 16 |
| Head+Tail \| Intermediate (17th) \| RAG | 0.595 | 0.033 | 128 | 5.5e-4 | 0.40 | 2.1e-3 | 64 |
| Head+Tail \| Intermediate (17th) \| No RAG | 0.588 | 0.033 | 256 | 2.2e-3 | 0.30 | 3.6e-4 | 64 |
| BumbaBERT \| Final (28th) \| RAG | 0.568 | 0.038 | 512 | 4.6e-4 | 0.40 | 1.2e-4 | 16 |
| BumbaBERT \| Intermediate (17th) \| RAG | 0.565 | 0.043 | 512 | 1.3e-4 | 0.50 | 2.3e-4 | 64 |
| BumbaBERT \| Intermediate (17th) \| No RAG | 0.558 | 0.032 | 512 | 1.1e-4 | 0.30 | 1.2e-4 | 16 |
| BumbaBERT \| Final (28th) \| No RAG | 0.545 | 0.021 | 512 | 4.2e-3 | 0.50 | 1.6e-3 | 16 |

### Linear Probing — Selected Hyperparameters

| Configuration | ma-F1 | Std | LR | Weight Decay | Batch Size |
|---|---|---|---|---|---|
| Head+Tail \| Intermediate (17th) \| RAG | 0.572 | 0.025 | 1.8e-3 | 2.1e-3 | 32 |
| Head+Tail \| Intermediate (17th) \| No RAG | 0.572 | 0.038 | 1.5e-3 | 1.5e-4 | 16 |
| Head+Tail \| Final (28th) \| No RAG | 0.571 | 0.023 | 4.0e-4 | 1.1e-5 | 16 |
| Head+Tail \| Final (28th) \| RAG | 0.556 | 0.017 | 1.8e-2 | 7.1e-5 | 128 |
| BumbaBERT \| Final (28th) \| RAG | 0.538 | 0.037 | 1.2e-4 | 4.4e-2 | 128 |
| BumbaBERT \| Intermediate (17th) \| RAG | 0.522 | 0.021 | 1.9e-4 | 5.9e-2 | 16 |
| BumbaBERT \| Intermediate (17th) \| No RAG | 0.520 | 0.032 | 1.7e-3 | 1.4e-2 | 64 |
| BumbaBERT \| Final (28th) \| No RAG | 0.485 | 0.020 | 2.1e-3 | 1.7e-5 | 16 |

## Notebooks

The `notebooks/` directory contains the training pipelines used to run both probing architectures across all 8 scenarios, including the hyperparameter search (Optuna/TPE), stratified 5-fold cross-validation, SMOTE-based class balancing (applied within training folds only), and early stopping on macro F1.

> **Note:** The notebooks were developed for execution on Google Colab and expect pre-extracted hidden-state activation files (`.parquet`) mounted from Google Drive. These activation files are not included in this repository due to the sensitive nature of the underlying judicial data, in accordance with the Lei Geral de Proteção de Dados Pessoais (LGPD).

## Data Availability

Due to LGPD and the sensitive nature of the judicial data used in this study, the original petition dataset and extracted hidden-state activations cannot be made publicly available. Requests for the experimental code or further details may be directed to the corresponding author.

## Citation

If you use this material, please cite the corresponding paper (citation details to be added upon publication).
