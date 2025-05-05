import numpy as np
import json
import os
import fire
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import sklearn.metrics
from scipy.sparse import csr_matrix  
import tqdm


# Define a TrainingSet structure 
class TrainingSet:
    def __init__(self, samples_by_class: dict):
        self.samples_by_class = samples_by_class

    def to_json(self, filename: str):
        # Ensure keys are strings for JSON serialization 
        data_to_save = {str(k): v for k, v in self.samples_by_class.items()}
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        # print(f"Selected training set saved to {filename}")

    def total_selected(self):
        return sum(len(ids) for ids in self.samples_by_class.values())

    def all_selected_ids(self):
        all_ids = set()
        for ids_list in self.samples_by_class.values():
            all_ids.update(ids_list)
        return all_ids


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# --- Selection Class (Medical TF-IDF Nested Cross-Validation Baseline) ---
class MedicalTFIDFSelection:
    def __init__(
        self,
        train_x_path: str, # Path to X_train_full_vec.npz
        train_y_path: str, # Path to y_train_full.npy
        train_ids_path: str, # Path to train_sample_ids.npy
        train_set_size_limit: int, # The total number of samples to select (target size for nested CV)
        random_seed: int = 42, # Random seed for reproducibility
        n_folds: int = 10, # Number of folds for nested cross-validation
        outer_split_size: float = 0.8, # Proportion of data for outer train split
    ):
        """
        Initializes the selection algorithm with preprocessed data.

        Args:
            train_x_path: Path to the preprocessed training features (.npz).
            train_y_path: Path to the preprocessed training labels (.npy).
            train_ids_path: Path to the original training sample IDs (.npy).
            train_set_size_limit: The maximum number of samples to select.
            random_seed: Random seed for selection.
            n_folds: Number of folds for nested cross-validation.
            outer_split_size: Proportion of data for the outer training fold.
        """
        self.train_x_path = train_x_path
        self.train_y_path = train_y_path
        self.train_ids_path = train_ids_path
        self.train_set_size_limit = train_set_size_limit
        self.random_seed = random_seed
        self.n_folds = n_folds
        self.outer_split_size = outer_split_size

        print("Loading preprocessed training data for selection...")
        try:
            self.X_train_full_vec = load_sparse_csr(train_x_path) # Load sparse matrix
            self.y_train_full = np.load(train_y_path) # Load labels
            self.train_sample_ids = np.load(train_ids_path) # Load original IDs
            print(f"Full Training data shape: {self.X_train_full_vec.shape}")

        except FileNotFoundError as e:
            print(f"Error loading preprocessed data for selection: {e}")
            raise

        if self.train_set_size_limit > len(self.y_train_full):
            print(f"Requested train_set_size_limit ({self.train_set_size_limit}) is larger than available data ({len(self.y_train_full)}). Using all available data.")
            self.train_set_size_limit = len(self.y_train_full)
        elif self.train_set_size_limit <= 0:
             raise ValueError("train_set_size_limit must be positive.")

        self.unique_labels = np.unique(self.y_train_full)
        print(f"Full training data contains {len(self.y_train_full)} samples across {len(self.unique_labels)} classes.")


    def select(self) -> TrainingSet:
        """
        Returns:
            TrainingSet: An object containing the selected sample IDs grouped by class.
        """

        print(f"\n--- Starting Medical TF-IDF Nested Cross-Validation Selection Baseline ---")
        print(f"Train set size limit (final selection): {self.train_set_size_limit}")
        print(f"Random seed: {self.random_seed}")
        print(f"Nested CV folds: {self.n_folds}")
        print(f"Outer split train proportion: {self.outer_split_size}")


        best_score = -1.0 
        best_outer_train_ixs = None
        best_inner_train_ixs = None


        # Outer Cross-Validation Loop (Stratified Shuffle Split on Full Data)
        crossfold_outer = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=self.n_folds,
            train_size=self.outer_split_size, # Proportion of data for the outer training set
            random_state=self.random_seed,
        )

        print("\nRunning Nested Cross-Validation:")
        for fold_idx, (outer_train_ixs, outer_val_ixs) in enumerate(tqdm.tqdm(
            crossfold_outer.split(self.X_train_full_vec, self.y_train_full),
            desc="Outer CV Fold",
            total=self.n_folds,
        )):
            # Subset the data for the current outer fold
            X_outer_train = self.X_train_full_vec[outer_train_ixs]
            y_outer_train = self.y_train_full[outer_train_ixs]
            # X_outer_val = self.X_train_full_vec[outer_val_ixs] 
            # y_outer_val = self.y_train_full[outer_val_ixs] 

            # Inner Cross-Validation Loop (Stratified Shuffle Split on Outer Train Data)
            # This splits the Outer Training data into Inner Train and Inner Validation folds.
            # The 'train_size' here is the *absolute number* of samples for the FINAL selection size limit
            # applied to the data available in the *current Outer Training fold*.
            inner_train_size_abs = self.train_set_size_limit

            # Ensure inner_train_size does not exceed the size of the current outer training set
            inner_train_size_abs = min(inner_train_size_abs, len(y_outer_train))

            # Skip inner CV if the outer train fold is too small to form the requested subset
            if inner_train_size_abs == 0 or len(y_outer_train) < inner_train_size_abs:
                 print(f"Outer train fold {fold_idx} ({len(y_outer_train)} samples) is too small to select {self.train_set_size_limit} samples. Skipping inner CV for this fold.")
                 continue

            # Create the Inner Splitter. train_size is the absolute number of samples.
            crossfold_inner = sklearn.model_selection.StratifiedShuffleSplit(
                n_splits=self.n_folds, # Same number of inner folds as outer
                train_size=inner_train_size_abs, # The NUMBER OF SAMPLES for the inner training set (our final selection size)
                random_state=self.random_seed + fold_idx, # Use a different seed per outer fold for inner splits
            )

            # Iterate through the inner folds (splits of the outer training data)
            inner_loop_iterator = crossfold_inner.split(X_outer_train, y_outer_train)

            for inner_fold_idx, (inner_train_ixs, inner_val_ixs) in enumerate(inner_loop_iterator):

                # Subset the data for the current inner fold
                X_inner_train = X_outer_train[inner_train_ixs]
                y_inner_train = y_outer_train[inner_train_ixs]
                X_inner_val = X_outer_train[inner_val_ixs]
                y_inner_val = y_outer_train[inner_val_ixs]

                # Train the fixed model 
                clf = sklearn.ensemble.VotingClassifier(
                    estimators=[
                        ("svm", sklearn.svm.SVC(probability=True, random_state=self.random_seed + fold_idx + inner_fold_idx)), # Use distinct seeds for inner models
                        ("lr", sklearn.linear_model.LogisticRegression(random_state=self.random_seed + fold_idx + inner_fold_idx, max_iter=1000)),
                    ],
                    voting="soft",
                )

                if X_inner_train.shape[0] > 0:
                    clf.fit(X_inner_train, y_inner_train)

                    # Evaluate on the Inner Validation data (using the model trained on Inner Train)
                    pred_y = clf.predict(X_inner_val)
                    score = sklearn.metrics.f1_score(y_inner_val, pred_y, average="macro")

                    # Check if this score is the best so far across all outer and inner folds
                    if score > best_score:
                        best_score = score
                        # inner_train_ixs are relative to outer_train_ixs
                        # outer_train_ixs are relative to the full data
                        best_outer_train_ixs = outer_train_ixs 
                        best_inner_train_ixs = inner_train_ixs 
                        # print(f"  -> New best score found: {best_score:.4f} (Outer {fold_idx}, Inner {inner_fold_idx})") 
                # else:
                    # print(f"Warning: Inner train fold ({X_inner_train.shape[0]} samples) is empty. Skipping training.") 

        #
        if best_outer_train_ixs is None or best_inner_train_ixs is None:
            selected_indices_full_data = np.array([], dtype=int) # Return empty selection
        else:
            # Get the indices relative to the full data for the best inner training split
            selected_indices_full_data = best_outer_train_ixs[best_inner_train_ixs]


        print(f"\nNested CV Selection complete. Best F1 Macro Score found during CV: {best_score:.4f}")
        print(f"Selected {len(selected_indices_full_data)} sample indices based on nested CV performance.")


        # Map selected indices back to original sample IDs and group by class
        selected_samples_by_class = {}
        if selected_indices_full_data.shape[0] > 0:
             # Get the original IDs and labels for the selected indices
             selected_original_ids = self.train_sample_ids[selected_indices_full_data]
             selected_labels = self.y_train_full[selected_indices_full_data]

             # Group by label
             for original_id, label in zip(selected_original_ids, selected_labels):
                 label_int = int(label) # Ensure label is int
                 if label_int not in selected_samples_by_class:
                     selected_samples_by_class[label_int] = []
                 selected_samples_by_class[label_int].append(int(original_id)) # Ensure ID is int for JSON

        # Create and return the TrainingSet object
        training_set = TrainingSet(
            samples_by_class=selected_samples_by_class,
        )

        print(f"Selected training set created with {training_set.total_selected()} samples.")
        if training_set.total_selected() > 0:
             for label in sorted(training_set.samples_by_class.keys()):
                  print(f"  Label {label}: {len(training_set.samples_by_class[label])} samples")


        return training_set


import fire

# This function will be called by fire to run the selection
def main(
    preprocessed_data_dir: str = "preprocessed_data/",
    output_file: str = "selected_train_ids.json", # Output file for selected IDs
    train_size: int = 100, # The desired total size of the selected training set (this is the target size of the subset chosen by nested CV)
    random_seed: int = 42, # Random seed for reproducibility
    n_folds: int = 10, # Number of folds for nested cross-validation
    outer_split_size: float = 0.8, # Proportion of data for outer train split in nested CV
):
    """
    Runs the medical TF-IDF nested cross-validation baseline selection algorithm.

    Args:
        preprocessed_data_dir: Directory containing the preprocessed .npz and .npy files.
        output_file: Path to save the selected sample IDs (JSON format).
        train_size: The total number of samples to select (this is the target size of the subset chosen by nested CV).
        random_seed: Random seed for selection.
        n_folds: Number of folds for nested cross-validation.
        outer_split_size: Proportion of data for the outer training fold in nested CV.
    """
    print("--- Medical TF-IDF Nested Cross-Validation Selection Script ---")

    train_x_path = os.path.join(preprocessed_data_dir, 'X_train_full_vec.npz')
    train_y_path = os.path.join(preprocessed_data_dir, 'y_train_full.npy')
    train_ids_path = os.path.join(preprocessed_data_dir, 'train_sample_ids.npy')

    selector = MedicalTFIDFSelection(
        train_x_path=train_x_path,
        train_y_path=train_y_path,
        train_ids_path=train_ids_path,
        train_set_size_limit=train_size,
        random_seed=random_seed,
        n_folds=n_folds,
        outer_split_size=outer_split_size
    )

    selected_set = selector.select()

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    selected_set.to_json(output_file)

    print(f"\nSelection script finished. Selected {selected_set.total_selected()} samples.")
    print(f"Selected IDs saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)