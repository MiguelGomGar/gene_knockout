import torch
import torch.nn as nn
from torch.utils.data import Dataset
import copy
import numpy as np
from sklearn.metrics import mean_squared_error

#%% Loss function
def custom_loss(y_pred, y_true, lambda_var=50.0):
    """
    Combines MSE and gene_var_score loss to penalize both mean and variance differences 
    between the true and predicted values.
    
    Args:
        - y_pred: Predicted values (PyTorch tensor).
        - y_true: True target values (PyTorch tensor).
        - lambda_var: Hyperparameter to control the weight of the variance penalty.
        
    Returns:
        - loss: The combined loss value.
    """

    mse_loss = nn.MSELoss()(y_pred, y_true)
    var_pred = torch.var(y_pred, dim=0)
    var_true = torch.var(y_true, dim=0)
    var_loss = torch.mean((var_pred - var_true)**2)
    
    return mse_loss + lambda_var * var_loss

class CustomVarianceLoss(nn.Module):
    """
    Loss function that combines the Mean Squared Error (MSE)
    with a penalty for the difference in variance gene by gene.
    """
    def __init__(self, lambda_var=50.0):
        super(CustomVarianceLoss, self).__init__()
        
        # Save the hyperparameter for the variance penalty as an attribute of the class
        self.lambda_var = lambda_var
        
        self.mse_criterion = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Compute the standard MSE loss
        mse_loss = self.mse_criterion(y_pred, y_true)
        
        # Penalty for difference in variance gene by gene
        var_pred = torch.var(y_pred, dim=0)
        var_true = torch.var(y_true, dim=0)
        var_loss = torch.mean((var_pred - var_true)**2)
        
        # Total loss
        return mse_loss + self.lambda_var * var_loss

#%% Data Loader
class GeneDataset(Dataset):
    """Wrapper for the list of dictionaries containing the data, to be used with PyTorch DataLoader"""
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]['diseased'], self.data[idx]['intervention'], self.data[idx]['treated']
    
#%% Training Loop
def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=15, patience=3):
    """
    Trains a PyTorch regression model with Early Stopping.
    
    Args:
        - model: The PyTorch model to train.
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data.
        - criterion: Loss function (e.g., CustomVarianceLoss or nn.MSELoss).
        - optimizer: PyTorch optimizer (e.g., Adam).
        - device: 'cuda' or 'cpu'.
        - epochs: Maximum number of epochs to train.
        - patience: Number of epochs to wait for improvement before stopping.
        
    Returns:
        model: The trained model with the best validation weights.
        history: Dictionary with 'train_loss' and 'val_loss' lists for plotting.
    """
    history = {'train_loss': [], 'val_loss': []}
    
    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for diseased, interv, treated in train_loader:
            diseased, interv, treated = diseased.to(device), interv.to(device), treated.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(diseased, interv)
            
            # Compute the loss
            loss = criterion(predictions, treated)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for diseased, interv, treated in val_loader:
                diseased, interv, treated = diseased.to(device), interv.to(device), treated.to(device)
                predictions = model(diseased, interv)
                
                loss = criterion(predictions, treated)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save the best weights in memory
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break
    
    # Restore the best model weights before returning
    if best_model_weights is not None:
        print(f"Restoring best model weights (Validation Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_weights)
        
    return model, history