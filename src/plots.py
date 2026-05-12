import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%% Scatter plot
def scatter_plot(x, y,
                title="Predicted vs Real Values", xlabel="Real Values", ylabel="Predicted Values"):
    """
    Displays a scatter plot of the x and y given values with a reference line.
    
    Args:
    - x: Array of x-values.
    - y: Array of y-values.
    - title: Title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    
    Returns:
    - Displays the scatter plot.
    """
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.3, s=15, color='steelblue')

    # y=x line
    max_val = max(np.max(x), np.max(y))
    plt.plot([0, max_val], [0, max_val], 'k-', label='y = x', linewidth=1)

    # Fit data to a line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, 'r-', label=f'fit: y={m:.3f}x + {b:.3f}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Improving the visualization
    plt.xlim(0, 0.75)
    plt.ylim(0, 0.75)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_losses(losses, 
                title="Training Loss Over Epochs",
                xlabel="Epoch", ylabel="Loss"):
    """
    Plots the training loss over epochs.
    
    Args:
    - losses: Dictionary containing 'train_loss' and 'val_loss' lists.
    - title: Title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    
    Returns:
    - Displays the loss plot.
    """
    
    # Get losses
    train_losses = losses['train_loss']
    val_losses = losses['val_loss']
    
    # Compute epoch numbers
    n_epochs = range(1, len(train_losses) + 1)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Line plots
    plt.plot(n_epochs, train_losses, marker='o', color='steelblue', label='Training')
    plt.plot(n_epochs, val_losses, marker='s', color='red', label='Validation')
    
    # Improving visualization
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()