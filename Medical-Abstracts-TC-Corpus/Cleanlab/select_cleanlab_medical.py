import numpy as np
import scipy.sparse
import json
import fire
from cleanlab_selection import SparseCleanlabSelection
import time # Added for timing

def load_sparse_matrix(filepath):
    """Load a sparse matrix from .npz file"""
    loader = np.load(filepath)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Removed n_jobs from main function signature
def main(
    preprocessed_data_dir: str = "preprocessed_data/",
    output_file: str = "Cleanlab/selected_train_ids.json",
    train_size: int = 2000,
    random_seed: int = 42,
    n_folds: int = 3, 
    class_balance: str = 'equal',
    extra_frac: float = 0.5, 
):
    """
    Args:
        preprocessed_data_dir: Directory containing preprocessed data files
        output_file: Where to save the selected training IDs
        train_size: Number of samples to select
        random_seed: Random seed for reproducibility
        n_folds: Number of cross-validation folds for cleanlab (reduced for speed)
        class_balance: How to balance classes ('equal' or 'proportional')
        extra_frac: Fraction of low-quality samples to remove additionally
    """
    start_load_time = time.time()
    X_train_full_vec = load_sparse_matrix(f"{preprocessed_data_dir}/X_train_full_vec.npz")
    y_train_full = np.load(f"{preprocessed_data_dir}/y_train_full.npy")
    train_sample_ids = np.load(f"{preprocessed_data_dir}/train_sample_ids.npy")
    print(f"Data loaded in {time.time() - start_load_time:.2f} seconds. Shape: {X_train_full_vec.shape}")

    # Initialize selector
    selector = SparseCleanlabSelection(
        X_train_full_vec=X_train_full_vec,
        y_train_full=y_train_full,
        train_sample_ids=train_sample_ids,
        train_set_size=train_size,
        random_seed=random_seed
    )

    # Run selection
    print(f"\nSelecting {train_size} samples using cleanlab with {n_folds} folds...")
    start_select_time = time.time()
    selected_indices, selected_ids = selector.select(
        class_balance=class_balance,
        num_folds=n_folds,
        extra_frac=extra_frac 

    )
    print(f"\nSelection process completed in {time.time() - start_select_time:.2f} seconds.")


    # Save results
    print(f"\nSaving selected IDs to {output_file}")
    start_save_time = time.time()
    with open(output_file, 'w') as f:
        # Convert numpy array to list for JSON serialization
        json.dump({"selected_ids": selected_ids.tolist()}, f)
    print(f"Saving completed in {time.time() - start_save_time:.2f} seconds.")


    print("\nDone! Selected samples distribution (Original Labels):")
    # Use the original full labels and the selected indices to show the final distribution
    unique, counts = np.unique(y_train_full[selected_indices], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

if __name__ == "__main__":
    fire.Fire(main)