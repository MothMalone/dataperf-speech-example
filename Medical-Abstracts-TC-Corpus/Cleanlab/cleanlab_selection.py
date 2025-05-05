from typing import Dict, List
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import tqdm
import scipy.sparse
import cleanlab
from cleanlab.classification import CleanLearning
from sklearn_extra.cluster import KMedoids
import time # Added for timing

class SparseCleanlabSelection:
    def __init__(self, X_train_full_vec, y_train_full, train_sample_ids, train_set_size, random_seed=42) -> None:
        """
        Parameters:
            X_train_full_vec: Sparse matrix of feature vectors
            y_train_full: Array of class labels
            train_sample_ids: Array of sample IDs
            train_set_size: Size of the final training set
            random_seed: Random seed for reproducibility
        """
        self.X_train_full_vec = X_train_full_vec
        self.y_train_full = y_train_full
        self.train_sample_ids = train_sample_ids
        self.train_set_size = train_set_size
        self.random_seed = random_seed

        # Convert 1-indexed labels to 0-indexed
        # Ensure unique values are sorted for consistent mapping
        unique_y = np.unique(y_train_full)
        self.label_map = {old: new for new, old in enumerate(unique_y)}
        self.reverse_label_map = {new: old for old, new in self.label_map.items()}
        self.y_train_full_zero_indexed = np.array([self.label_map[y] for y in y_train_full])

    def select(self, class_balance='equal', num_folds=10, extra_frac=0.5):
        """
        Parameters:
            class_balance: 'equal' - Same number of samples per class
                          'proportional' - Keep the same proportions as original dataset
            num_folds: Number of cross-validation folds for cleanlab
            extra_frac: Fraction of datapoints to delete with lowest cleanlab label quality scores (beyond default cleanlab filter)
        
        Returns:
            selected_indices: Indices of selected samples
            selected_ids: IDs of selected samples
        """
        NUM_FOLDS = num_folds
        EXTRA_FRAC = extra_frac

        SEED = self.random_seed

        # --- Optimization 1: Use Logistic Regression ---
        # Removed SVC for significant speedup.
        clf = sklearn.linear_model.LogisticRegression(
             solver='liblinear', # 'liblinear' often good for small datasets or sparse L1/L2
             random_state=SEED

             )

        # Identify all unique classes (using 0-indexed)
        unique_classes_zero_indexed = np.unique(self.y_train_full_zero_indexed)
        num_classes = len(unique_classes_zero_indexed)
        
        # Calculate per-class samples based on balance strategy
        # Use 0-indexed labels for internal calculations
        if class_balance == 'equal':
            # Equal number of samples per class
            per_class_size = self.train_set_size // num_classes
            class_sizes_zero_indexed = {cls_zero: per_class_size for cls_zero in unique_classes_zero_indexed}
        else:  # proportional
            # Maintain class proportions
            class_counts = {cls_zero: np.sum(self.y_train_full_zero_indexed == cls_zero) for cls_zero in unique_classes_zero_indexed}
            total_samples = len(self.y_train_full_zero_indexed)
            class_sizes_zero_indexed = {
                cls_zero: max(1, int(self.train_set_size * (count / total_samples))) 
                for cls_zero, count in class_counts.items()
            }
            # Adjust to ensure we get exactly train_set_size samples
            total_allocated = sum(class_sizes_zero_indexed.values())
            if total_allocated < self.train_set_size:
                # Distribute remaining samples to largest classes
                sorted_classes = sorted(
                    [(cls_zero, count) for cls_zero, count in class_counts.items()],
                    key=lambda x: x[1], reverse=True
                )
                remaining = self.train_set_size - total_allocated
                for i in range(remaining):
                     # Add to the classes with most samples first, cycling if needed
                     # Use the original label for printing, but update using zero-indexed key
                    cls_to_add = sorted_classes[i % len(sorted_classes)][0]
                    class_sizes_zero_indexed[cls_to_add] += 1

        print(f"Number of classes: {num_classes}")
        print("Class distribution target (0-indexed labels):")
        for cls_zero, size in class_sizes_zero_indexed.items():
             print(f"  Class {self.reverse_label_map[cls_zero]}: {size} samples (0-indexed: {cls_zero})")

        print("Using cleanlab + K-medoids clustering to select data subset ...")
        
        # --- Optimization 2: Reduce folds  ---
        start_time = time.time()
        cl = cleanlab.classification.CleanLearning(
            clf, seed=SEED, verbose=True, cv_n_folds=NUM_FOLDS,
            label_quality_scores_kwargs={"adjust_pred_probs": True},
        )
        
        # working with the whole dataset at once
        Xs = self.X_train_full_vec
        ys = self.y_train_full_zero_indexed
        
        # Find label issues
        # Removed reference to n_jobs in print statement
        print(f"Finding label issues using Cleanlab with {NUM_FOLDS} folds...")
        issues = cl.find_label_issues(Xs, ys)
        print(f"Cleanlab find_label_issues completed in {time.time() - start_time:.2f} seconds.")

        bad_indices_filter = issues[issues['is_label_issue']].index.values # Get indices directly from the dataframe
        
        if EXTRA_FRAC > 0:
            num_delete_cleanlab = int(EXTRA_FRAC * len(ys))
            # Sort by label quality score (lowest first) and get the top num_delete_cleanlab indices
            bad_indices_rank = issues.nsmallest(num_delete_cleanlab, 'label_quality').index.values
            bad_indices = np.unique(np.concatenate((bad_indices_filter, bad_indices_rank)))
        else:
            bad_indices = bad_indices_filter
            
        # Ensure bad_indices are within the original indices range
        all_original_indices = np.arange(len(self.y_train_full))
        inds_to_keep_mask = np.ones(len(self.y_train_full), dtype=bool)
        inds_to_keep_mask[bad_indices] = False
        inds_to_keep = all_original_indices[inds_to_keep_mask]


        # Cluster remaining data:
        # Store original mapping before slicing
        original_indices_mapping = {new_idx: original_idx for new_idx, original_idx in enumerate(inds_to_keep)}

        ys_filtered = ys[inds_to_keep]
        Xs_filtered = Xs[inds_to_keep]
        
        print("Class distribution after cleanlab filtering: ")
        unique_filtered, counts_filtered = np.unique([self.reverse_label_map[y_zero] for y_zero in ys_filtered], return_counts=True)
        print(np.asarray((unique_filtered, counts_filtered)).T)
        print(f"Keeping {len(inds_to_keep)} samples after filtering label issues and low quality scores.")

        print("Finding coresets among remaining clean data using K-Medoids ...")
        start_time = time.time()
        best_indices_in_filtered = np.array([], dtype=int) # Indices within the filtered arrays (ys_filtered, Xs_filtered)
        
        for y_zero in unique_classes_zero_indexed:
            y = self.reverse_label_map[y_zero] # Original label for printing
            target_per_class_size = class_sizes_zero_indexed[y_zero]
            
            if target_per_class_size == 0:
                continue
                
            # Get indices within the *filtered* data for this class
            y_inds_in_filtered = np.where(ys_filtered == y_zero)[0]
            
            num_available_samples = len(y_inds_in_filtered)
            print(f"Finding coreset for class {y} (0-indexed: {y_zero}). Target size: {target_per_class_size}, Available after filter: {num_available_samples} ...")

            # Handle case where we have fewer samples than requested per_class_size
            actual_clusters = min(target_per_class_size, num_available_samples)
            if actual_clusters <= 0: # Ensure we cluster only if needed and possible
                print(f"  No samples or target size is zero for class {y}, skipping K-Medoids.")
                continue
            
            Xs_y_filtered = Xs_filtered[y_inds_in_filtered]
            
            # --- Bottleneck 2: Conversion to Dense ---
            # This is still necessary because KMedoids doesn't support sparse.
            # Could have memory usage here. If this fails, might need
            # dimensionality reduction or a different clustering algorithm.
            start_dense_time = time.time()
            if scipy.sparse.issparse(Xs_y_filtered):
                 print(f"  Converting sparse data for class {y} (shape {Xs_y_filtered.shape}) to dense...")
                 Xs_y_dense = Xs_y_filtered.toarray()
                 print(f"  Dense conversion completed in {time.time() - start_dense_time:.2f} seconds.")
            else: # Should already be dense if it wasn't sparse to begin with
                 Xs_y_dense = Xs_y_filtered

            # --- KMedoids Clustering  ---
            start_cluster_time = time.time()
            try:
                clstr = KMedoids(n_clusters=actual_clusters, init="k-medoids++", random_state=SEED)
                clstr.fit(Xs_y_dense)
                coreset_y_inds_in_filtered_class = clstr.medoid_indices_
                print(f"  K-Medoids fitting completed in {time.time() - start_cluster_time:.2f} seconds.")

                # Map the indices back to the global filtered indices
                global_filtered_y_inds = y_inds_in_filtered[coreset_y_inds_in_filtered_class]
                best_indices_in_filtered = np.concatenate((best_indices_in_filtered, global_filtered_y_inds))

            except ValueError as e:
                print(f"  Error during K-Medoids for class {y}: {e}")
                # If clustering fails (e.g., not enough unique points for clusters),
                # you might need a fallback. For simplicity here, we skip the class.
                # A robust solution might select a random subset instead.
                pass

        # Map the selected indices within the filtered data back to the original indices
        selected_indices = np.array([original_indices_mapping[idx] for idx in best_indices_in_filtered])

        print(f"K-Medoids clustering for all classes completed in {time.time() - start_time:.2f} seconds.")
        
        print("Class distribution post-selection (0-indexed labels): ")
        # Use the original full labels to check distribution of selected indices
        selected_ys_zero_indexed = ys[selected_indices]
        unique_selected, counts_selected = np.unique(selected_ys_zero_indexed, return_counts=True)

        # Print with original labels for clarity
        selected_distribution_original = {}
        for u_zero, c in zip(unique_selected, counts_selected):
            selected_distribution_original[self.reverse_label_map[u_zero]] = c

        # Print for all target classes, even if none were selected
        print("Selected Class Distribution (Original Labels):")
        for cls_zero in unique_classes_zero_indexed:
             original_label = self.reverse_label_map[cls_zero]
             count = selected_distribution_original.get(original_label, 0)
             target_size = class_sizes_zero_indexed[cls_zero]
             print(f"  Class {original_label}: {count} selected (Target: {target_size})")


        print(f"Selected {len(selected_indices)} samples in total.")

        selected_ids = self.train_sample_ids[selected_indices]
        
        return selected_indices, selected_ids

# Example usage:
# selector = SparseCleanlabSelection(
#     X_train_full_vec=X_train_full_vec,
#     y_train_full=y_train_full,
#     train_sample_ids=train_sample_ids,
#     train_set_size=5000,
#     random_seed=42
# )
# selected_indices, selected_ids = selector.select(class_balance='equal', num_folds=3, extra_frac=0.1)