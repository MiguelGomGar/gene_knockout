import torch
import numpy as np

def reduce_dimensionality_hvg(dataset, top_k=1000):
    """
    Reduces the dimensionality of the dataset by selecting the 'top_k' genes 
    with the highest variance between the 'diseased' and 'intervention' states.
    
    Args:
        - dataset: A list of dictionaries, where each dictionary contains 'diseased', 'intervention', and 'treated' tensors.
        - top_k: The number of top genes to select based on variance.

    Returns: 
        - reduced_dataset: A new list of dictionaries with only the top_k genes.
        - top_indices: The indices of the selected genes in the original dataset.
    """
    
    # Stack all diseased samples to calculate global variance
    all_diseased = torch.stack([s['diseased'] for s in dataset])
    all_intervention = torch.stack([s['intervention'] for s in dataset])
    
    # Calculate variance for each gene
    gene_variances = torch.var(all_diseased - all_intervention, dim=0)
    
    # Get the indices of the top_k genes with highest variance
    _, top_indices = torch.topk(gene_variances, top_k)
    
    # Sort indices just to keep the original relative order
    top_indices, _ = torch.sort(top_indices)
    
    # Create a new dataset keeping only these indices
    reduced_dataset = []
    for s in dataset:
        reduced_sample = {
            'diseased': s['diseased'][top_indices],
            'intervention': s['intervention'][top_indices],
            'treated': s['treated'][top_indices],
            
            # If gene_symbols is a list, we filter it too
            'gene_symbols': [s['gene_symbols'][i] for i in top_indices.tolist()] if 'gene_symbols' in s else None
        }
        reduced_dataset.append(reduced_sample)
            
    return reduced_dataset, top_indices