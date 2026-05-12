# Deep Learning for Genetic Perturbation Prediction

This repository contains the final project for the Deep Learning course, focused on predicting the cellular response to genetic perturbations (CRISPR) using gene expression data.

## Project Overview

The goal of this project is to develop a computational model capable of predicting the expression state of a cell after a specific genetic intervention (`treated` state), given its initial `diseased` state and the `intervention` vector (one-hot encoded).

## Methodology and Model Optimization

The project followed a structured optimization path, starting from a simple baseline to advanced deep learning architectures:

1.  **Baseline Model**: Implementation of a Ridge Regression model to establish a performance floor. This model was used to evaluate different splitting strategies (Random vs. Unseen Interventions).
2.  **Original MLP**: A dual-encoder architecture that processes the diseased state and the intervention vector separately before concatenating them for the final prediction.
3.  **Hyperparameter Tuning**: 
    * **Learning Rate**: Systematic evaluation of different learning rates ($10^{-4}, 10^{-5}, 10^{-6}$) to find the optimal convergence speed and stability.
    * **Dimensionality Reduction**: Testing the effect of using the full gene set (10,716 genes) versus reducing to Highly Variable Genes (HVG) using variance-based selection.
4.  **Architectural Improvements (SkipConn MLP)**:
    * **Batch Normalization**: Added to each layer to stabilize training and improve gradient flow.
    * **Residual Connections (Skip Connections)**: Modified the output to predict the *change* (delta) rather than the absolute state, significantly improving variance retention.
    * **Custom Loss Function**: Implementation of a variance-penalized loss function to ensure the model preserves the biological distribution of gene expression.
5.  **Cross-Cell-Line Validation**: Evaluation of the model's ability to generalize by training on one or two cell lines (e.g., A549, HT29) and testing on a completely unseen cell line (PC3).

## Results

Detailed performance metrics and visualizations can be found in the `results/` folder.

### Metrics Summary (Excel File)
The `metrics.xlsx` file provides a comprehensive breakdown of every optimization step. Each tab corresponds to a specific experiment:
* **Splitting Mode**: Comparison of results between random and unseen intervention splits for the baseline.
* **Learning Rate**: Performance of the original MLP across different learning rates.
* **Dim Reduct**: Evaluation of the impact of reducing dimensionality on the original MLP.
* **Loss Function**: Comparison between standard MSE and the Custom Variance-Penalized Loss.
* **Modifications (SkipCon MLP)**: Final comparison of the architectural improvements (Residuals, BatchNorm, Dropout).

### Visualizations
The folder also contains plots for:
* Training/Validation loss curves.
* Scatter plots of predicted vs. true standard deviation per gene (Variance preservation analysis).

## Repository Structure

* `notebooks/`: Jupyter notebooks for data generation and the main training pipeline.
* `src/`: Modular Python scripts containing the logic for preprocessing, model architectures, training, and evaluation.
* `results/`: All output metrics and plots.
* `data/`: (Not included in repo) Contains the raw `.pt` datasets.

## Requirements

* Python 3.x
* PyTorch
* Scikit-Learn
* NumPy
* Matplotlib
* Pandas
