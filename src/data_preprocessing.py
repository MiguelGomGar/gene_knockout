import os
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def load_dataset_from_torch_data(cell_line, path_root):
    """
    Loads the normalized dataset from a .pt file (data_backward_{cell_line}.pt),
    converts each sample to tensors, and transforms it into a dictionary with
    keys 'diseased', 'intervention', 'treated' and 'gene_symbols'. Returns a
    list of samples in that format, ready for training/evaluation.
    """
    
    file_path = os.path.join(path_root, f"data_backward_{cell_line}.pt")
    loaded_data = torch.load(file_path, map_location='cpu', weights_only=False)
    dataset = []
    
    for d in loaded_data:
        diseased = d.diseased.float() if torch.is_tensor(d.diseased) else torch.tensor(d.diseased).float()
        intervention = d.intervention.float() if torch.is_tensor(d.intervention) else torch.tensor(d.intervention).float()
        treated = d.treated.float() if torch.is_tensor(d.treated) else torch.tensor(d.treated).float()
        
        gene_symbols = d.gene_symbols
        dataset.append({'diseased': diseased, 
                        'intervention': intervention, 
                        'treated': treated, 
                        'gene_symbols': gene_symbols})
        
    return dataset 

def get_cached_dataset(cell_line, path_root):
    """
    Loads the processed dataset from a cache file if it exists.
    If it does not exist, it processes the raw .pt file using 
    load_dataset_from_torch_data, saves the processed list as a cache, 
    and returns it.
    
    Args:
        - cell_line: The name of the cell line's file.
        - path_root: The root directory where the .pt files are located.
    
    Returns:
        - dataset: A list of dictionaries with keys 'diseased', 'intervention', 'treated', and 'gene_symbols'.
    """
    cache_path = os.path.join(path_root, f"cached_dataset_{cell_line}.pt")
    
    dataset = torch.load(cache_path, weights_only=False)
    
    return dataset

def prep_sklearn_data(samples):
    """
    Converts a list of sample dictionaries into NumPy arrays suitable for 
    scikit-learn models (e.g., Linear Regression, Ridge, Random Forest).
    
    It concatenates the 'diseased' state and the 'intervention' one-hot vector 
    to form the input feature matrix X, and uses the 'treated' state as the 
    target matrix Y.
    
    Args:
    - samples: List of dictionaries. Each dictionary must contain:
        * 'diseased': torch.Tensor of shape (d,)
        * 'intervention': torch.Tensor of shape (d,)
        * 'treated': torch.Tensor of shape (d,)
        
    Returns:
    - X: np.ndarray of shape (n_samples, 2*d) containing the input features.
    - Y: np.ndarray of shape (n_samples, d) containing the target features.
    """
    X = np.array([torch.cat([s['diseased'], s['intervention']]).cpu().numpy() for s in samples])
    Y = np.array([s['treated'].cpu().numpy() for s in samples])
    
    return X, Y

def prepare_splits(dataset, val_size, test_size, splitting_scheme, N_folds=1, seed=42):
    """
    Generates train/val/test indices for a dataset with two schemes:
        - `random`: classic random split.
        - `unseen_interventions`: ensures interventions (perturbed genes)
    are disjoint between train/val/test.
    
    Returns:
        - train_idx_dict: dict {fold: [idx]} with train indices per fold.
        - val_idx_dict: dict {fold: [idx]} with validation indices per fold.
        - test_idx: list of test indices.
    """

    if splitting_scheme == "random":
        # Random classic split
        all_idx = list(range(len(dataset)))
        trainval_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed)
        rel_val_size = val_size / (1.0 - test_size)
        train_idx_dict, val_idx_dict = {}, {}
        
        for fold in range(N_folds):
            train_idx, val_idx = train_test_split(trainval_idx, test_size=rel_val_size, random_state=seed + fold)
            train_idx_dict[fold] = list(train_idx)
            val_idx_dict[fold] = list(val_idx)
            
        return train_idx_dict, val_idx_dict, list(test_idx)

    elif splitting_scheme == "unseen_interventions":
        # Train/Val/Test has unique interventions
        # Group by intervention
        group_keys = [tuple(np.nonzero(item['intervention'].cpu().numpy())[0].tolist()) for item in dataset]
        unique_map = {k: i for i, k in enumerate(dict.fromkeys(group_keys))}
        groups_int = np.array([unique_map[k] for k in group_keys], dtype=int)
        all_idx = list(range(len(dataset)))
        X = np.arange(len(all_idx))
        
        # test
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, test_idx = next(gss_test.split(X=X, groups=groups_int))
        test_idx = list(test_idx)
        pool_index = [i for i in all_idx if i not in set(test_idx)]
        rel_val_size = val_size / (1.0 - test_size)
        pool_groups = groups_int[pool_index]
        X_pool = np.arange(len(pool_index))
        
        # Train and validation
        train_idx_dict, val_idx_dict = {}, {}
        
        for fold in range(N_folds):
            gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed + fold)
            train_rel, val_rel = next(gss_val.split(X=X_pool, groups=pool_groups))
            train_idx = [pool_index[i] for i in train_rel]
            val_idx = [pool_index[i] for i in val_rel]
            train_idx_dict[fold] = list(train_idx)
            val_idx_dict[fold] = list(val_idx)
            
        return train_idx_dict, val_idx_dict, list(test_idx)

    else:
        raise ValueError("splitting_scheme must be 'random' or 'unseen_interventions'")

def apply_gene_filter(dataset, indices):
    """
    Filters a dataset to keep only the specified genes.
    """
    reduced_dataset = []
    for s in dataset:
        reduced_sample = {
            'diseased': s['diseased'][indices],
            'intervention': s['intervention'][indices],
            'treated': s['treated'][indices],
            'gene_symbols': [s['gene_symbols'][i] for i in indices.tolist()] if 'gene_symbols' in s else None
        }
        reduced_dataset.append(reduced_sample)
    return reduced_dataset