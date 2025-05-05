import numpy as np
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import sklearn.metrics
import tqdm
import yaml
import fire
import os
import json
from scipy.sparse import csr_matrix 




def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def main(
    preprocessed_data_dir: str = "preprocessed_data/",
    selected_ids_json: str = None, # Path to selected training IDs (output of select script)
    metrics_output_file: str = "metrics.json", # Output file for evaluation results
    use_full_training: bool = False, # New flag to force using full training set
):
    """
    Evaluates a fixed model trained on a SELECTED subset of preprocessed medical abstracts.

    Args:
        preprocessed_data_dir: Directory containing the preprocessed .npz and .npy files.
        selected_ids_json: Path to a JSON file containing the IDs of selected training samples.
                           If None, uses the full training dataset for training.
        metrics_output_file: Path to save the evaluation metrics (JSON format).
    """
    print("--- Medical Abstract Evaluation Script ---")

    try:
        X_train_full_vec = load_sparse_csr(os.path.join(preprocessed_data_dir, 'X_train_full_vec.npz'))
        y_train_full = np.load(os.path.join(preprocessed_data_dir, 'y_train_full.npy'))
        train_sample_ids = np.load(os.path.join(preprocessed_data_dir, 'train_sample_ids.npy')) # Load original IDs

        # Load test data (features, labels)
        X_test_vec = load_sparse_csr(os.path.join(preprocessed_data_dir, 'X_test_vec.npz'))
        y_test = np.load(os.path.join(preprocessed_data_dir, 'y_test.npy'))

        # vectorizer = joblib.load(os.path.join(preprocessed_data_dir, 'tfidf_vectorizer.joblib'))

        print(f"Full Training data shape: {X_train_full_vec.shape}, Labels shape: {y_train_full.shape}, IDs shape: {train_sample_ids.shape}")
        print(f"Test data shape: {X_test_vec.shape}, Labels shape: {y_test.shape}")

    except FileNotFoundError as e:
        print(f"Error loading preprocessed data: {e}")
        return 


    eval_random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Default seeds
    model_params = {} # Default empty params
   


    if use_full_training:
        print("\n--- Using Full Training Data (--use_full_training flag set) ---")
        X_train_selected_vec = X_train_full_vec
        y_train_selected = y_train_full
    elif selected_ids_json:
        print(f"\n--- Loading Selected Training IDs from {selected_ids_json} ---")
        try:
            with open(selected_ids_json, 'r') as f:
                selected_data_structure = json.load(f)

            # Convert the existing format to expected format
            if not ('targets' in selected_data_structure or 'nontargets' in selected_data_structure):
                reformatted_structure = {
                    "targets": {
                        str(label): ids for label, ids in selected_data_structure.items()
                    },
                    "nontargets": []
                }
                selected_data_structure = reformatted_structure

            # Extract all selected IDs into a single set for easy lookup
            selected_ids_set = set()
            if 'targets' in selected_data_structure:
                for label_ids_list in selected_data_structure['targets'].values():
                    selected_ids_set.update(label_ids_list)
            if 'nontargets' in selected_data_structure:
                selected_ids_set.update(selected_data_structure['nontargets'])
            # Filter the full preprocessed data based on selected IDs
            # Find the indices in the *full* training data arrays (based on original IDs)
            selected_indices = [
                i for i, sample_id in enumerate(train_sample_ids) # Iterate through original IDs
                if sample_id in selected_ids_set # Check if this ID was selected
            ]

            # Ensure all selected IDs were found in the full data
            if len(selected_indices) != len(selected_ids_set):
                print(f"Found {len(selected_indices)} matching IDs in full data, but {len(selected_ids_set)} selected IDs were listed in the JSON.")

            # Use the selected indices to subset the full preprocessed training data
            X_train_selected_vec = X_train_full_vec[selected_indices]
            y_train_selected = y_train_full[selected_indices]

            print(f"Filtered training data using selected subset. Shape: {X_train_selected_vec.shape}")

        except FileNotFoundError:
            print(f"Error: Selected IDs file not found at {selected_ids_json}. Using full training data for training.")
            # Fall back to using full data if selected file not found
            X_train_selected_vec = X_train_full_vec
            y_train_selected = y_train_full
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {selected_ids_json}. Using full training data for training.")
            X_train_selected_vec = X_train_full_vec
            y_train_selected = y_train_full
        except Exception as e:
            print(f"An error occurred loading selected IDs: {e}. Using full training data for training.")
            X_train_selected_vec = X_train_full_vec
            y_train_selected = y_train_full
    else:
        print("\n--- No selected_ids_json provided. Using Full Training Data for training ---")
        # If no selected file is provided, use the full data for training
        X_train_selected_vec = X_train_full_vec
        y_train_selected = y_train_full


    scores = [] # To store F1 scores
    
    if use_full_training:
        print("\n--- Training and Evaluating Once on Full Training Set ---")
        clf = sklearn.ensemble.VotingClassifier(
            estimators=[
                ("svm", sklearn.svm.SVC(probability=True, **model_params.get('svm', {}))),
                ("lr", sklearn.linear_model.LogisticRegression(max_iter=1000, **model_params.get('lr', {}))),
            ],
            voting="soft",
        )
        clf.fit(X_train_selected_vec, y_train_selected)
        pred_y = clf.predict(X_test_vec)
        score = sklearn.metrics.f1_score(y_test, pred_y, average="macro")
        scores.append(score)
        print(f"Single Run F1 Macro Score: {score:.4f}")
    else:
        print("\n--- Training and Evaluating with Multiple Random Seeds ---")
        # Original code with random seeds
        for i, random_seed in enumerate(tqdm.tqdm(eval_random_seeds, desc="Evaluating Seeds")):
            clf = sklearn.ensemble.VotingClassifier(
                estimators=[
                    ("svm", sklearn.svm.SVC(probability=True, random_state=random_seed, **model_params.get('svm', {}))),
                    ("lr", sklearn.linear_model.LogisticRegression(random_state=random_seed, max_iter=1000, **model_params.get('lr', {}))),
                ],
                voting="soft",
            )

            if X_train_selected_vec.shape[0] == 0:
                 print(f"Warning: Selected training subset is empty for seed {random_seed}. Skipping training.")
                 # Append NaN if no training samples
                 scores.append(np.nan) 
                 continue

            clf.fit(X_train_selected_vec, y_train_selected)

            pred_y = clf.predict(X_test_vec)

            score = sklearn.metrics.f1_score(y_test, pred_y, average="macro")
            scores.append(score)
            # print(f"Seed {random_seed} F1 Macro Score: {score:.4f}") 

    print("\n--- Evaluation Results ---")
    if scores:
        # Filter out NaNs
        valid_scores = [score for score in scores if not np.isnan(score)]
        if valid_scores:
            average_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            print(f"Individual F1 Macro Scores: {scores}")
            print(f"Average F1 Macro Score: {average_score:.4f}")
            print(f"Standard Deviation of F1 Macro Scores: {std_score:.4f}")

            metrics = {
                'average_f1_macro': float(average_score), # Convert numpy float to standard float for JSON
                'std_f1_macro': float(std_score),       # Convert numpy float to standard float
                'individual_scores': [float(s) if not np.isnan(s) else None for s in scores], # Convert to standard float or None
                'num_training_samples_used': int(X_train_selected_vec.shape[0]) if scores else 0 # Number of samples in the *last* training run
            }
            if selected_ids_json:
                metrics['num_selected_ids_listed'] = len(selected_ids_set)


            try:
                metrics_output_dir = os.path.dirname(metrics_output_file)
                if metrics_output_dir and not os.path.exists(metrics_output_dir):
                    os.makedirs(metrics_output_dir, exist_ok=True)

                with open(metrics_output_file, 'w') as f:
                     json.dump(metrics, f, indent=4)
                print(f"Metrics saved to {metrics_output_file}")
            except Exception as e:
                 print(f"Error saving metrics to {metrics_output_file}: {e}")

    else:
        print("No scores were calculated.")


if __name__ == "__main__":
    fire.Fire(main)