#%% Modules
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error

#%% Evaluation Metrics
def samplewise_r2(y_true, y_pred):
    """
    Computes sample-wise R^2 using the Pearson correlation between y_true[i] and y_pred[i].
    Input: y_true, y_pred array-like (lists or np.ndarray) with shape (n_samples, d).
    Output: np.ndarray of size n_samples with the R^2 per sample.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = y_true.shape[0]
    r2 = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        r2[i] = pearsonr(y_true[i], y_pred[i])[0] ** 2

    return r2

def gene_var_score(y_true, y_pred, eps=1e-12):
    """
    Variance score per gene based on log-variance ratio: score = exp(-|log((var_pred + eps)/(var_true + eps))|).
    Input: y_true, y_pred array-like (lists or np.ndarray) with shape (n_samples, d).
    Output: np.ndarray of size n_genes with the variance score for each gene.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    var_true = np.var(y_true, axis=0)
    var_pred = np.var(y_pred, axis=0)
    ratio = (var_pred + eps) / (var_true + eps)
    score = np.exp(-np.abs(np.log(ratio)))

    return score

#%% Models evaluation functions
def evaluate_baseline_model(model, X_test, Y_test):
    """
    Evaluates the baseline model on the test set by computing the required metrics.
    
    Args:
        - model: The trained baseline model to evaluate.
        - X_test: The input features for the test set.
        - Y_test: The true target values for the test set.
        
    Returns:
        - mse: Mean Squared Error of the predictions.
        - r2_scores: Average sample-wise R^2 score across all samples.
        - gene_var_scores: Average variance score across all genes.
    """
    Y_pred = model.predict(X_test)

    # Compute metrics
    mse = mean_squared_error(Y_test, Y_pred)
    r2_scores = samplewise_r2(Y_test, Y_pred).mean()
    gene_var_scores = gene_var_score(Y_test, Y_pred).mean()
    
    return mse, r2_scores, gene_var_scores

def evaluate_pytorch_model(model, test_loader, device='cuda'):
    """
    Evaluates the trained model on the test set and computes the required metrics.
    
    Args:
        - model: The trained PyTorch model to evaluate.
        - test_loader: The data loader for the test set.
        - device: The device to use for evaluation ('cuda' or 'cpu').

    Returns:
        - mse: Mean Squared Error of the predictions.
        - r2_scores: Average sample-wise R^2 score across all samples.
        - gene_var_scores: Average variance score across all genes.
        - Y_true_dl: True target values.
        - Y_pred_dl: Predicted target values.
    """
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for diseased, interv, treated in test_loader:
            diseased, interv = diseased.to(device), interv.to(device)
            
            preds = model(diseased, interv)
            
            all_preds.append(preds.cpu().numpy())
            all_trues.append(treated.numpy())

    Y_pred_dl = np.vstack(all_preds)
    Y_true_dl = np.vstack(all_trues)

    # Compute metrics
    mse = mean_squared_error(Y_true_dl, Y_pred_dl)
    r2_scores = samplewise_r2(Y_true_dl, Y_pred_dl).mean()
    gene_var_scores = gene_var_score(Y_true_dl, Y_pred_dl).mean()
    
    return mse, r2_scores, gene_var_scores, Y_true_dl, Y_pred_dl