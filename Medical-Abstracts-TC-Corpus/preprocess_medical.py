import pandas as pd
import numpy as np
import sklearn.feature_extraction.text
import tqdm
import os
import sys
import string
import re
import nltk
from collections import Counter
import joblib # To save the fitted vectorizer



def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def setup_nltk_data(download_dir=None):
    if download_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            download_dir = os.path.join(script_dir, "nltk_data")
        except NameError: 
             download_dir = os.path.join(os.getcwd(), "nltk_data")

        if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
             venv_nltk_dir = os.path.join(sys.prefix, 'nltk_data')
             if os.path.exists(venv_nltk_dir) or can_create_dir(venv_nltk_dir):
                 download_dir = venv_nltk_dir

    try:
        os.makedirs(download_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating NLTK download directory {download_dir}: {e}.")
        return 

    if download_dir not in nltk.data.path:
         nltk.data.path.insert(0, download_dir) 

    required_resources = ['punkt', 'stopwords']
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
             try:
                 nltk.download(resource, download_dir=download_dir, quiet=True)
                 print(f"-> '{resource}' download attempted.")
             except Exception as e:
                 print(f"-> Error during NLTK '{resource}' download to {download_dir}: {e}. Check directory permissions.")

    # print(f"\nFinal NLTK data search path: {nltk.data.path}")

    def can_create_dir(path):
         try:
             test_dir = os.path.join(path, '__test_write__')
             os.makedirs(test_dir, exist_ok=True)
             os.rmdir(test_dir)
             return True
         except Exception:
             return False


try:
    import nltk
    setup_nltk_data()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    print("NLTK data (punkt, stopwords) is ready.")
except ImportError:
    print("NLTK not installed. Skipping NLTK-based cleaning.")
    stop_words = set()
except LookupError:
     print("NLTK data not fully set up. Stop words might be incomplete.")
     stop_words = set()



import fire 

def main(
    input_dir: str = "",
    output_dir: str = "preprocessed_data/",
    train_file: str = "medical_tc_train.csv",
    test_file: str = "medical_tc_test.csv",
    max_features: int = 10000 
):
    """
    Args:
        input_dir: Directory containing the raw CSV files.
        output_dir: Directory to save the preprocessed .npz and .npy files.
        train_file: Name of the training CSV file in input_dir.
        test_file: Name of the testing CSV file in input_dir.
        max_features: Maximum number of features for TfidfVectorizer.
    """


    train_csv_path = os.path.join(input_dir, train_file)
    test_csv_path = os.path.join(input_dir, test_file)

  
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    print(f"Data loaded: {len(train_df)} training samples, {len(test_df)} testing samples.")


    train_sample_ids = train_df.index.values 

    X_train_full = train_df["medical_abstract"].values
    y_train_full = train_df["condition_label"].values
    X_test = test_df["medical_abstract"].values
    y_test = test_df["condition_label"].values




    X_train_full_cleaned = [clean_text(text) for text in tqdm.tqdm(X_train_full, desc="Cleaning Train Data")]
    X_test_cleaned = [clean_text(text) for text in tqdm.tqdm(X_test, desc="Cleaning Test Data")]



    print("\nFitting TF-IDF vectorizer...")

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=max_features,
        stop_words=list(stop_words) if stop_words else 'english',
    )

    X_train_full_vec = vectorizer.fit_transform(X_train_full_cleaned)
    X_test_vec = vectorizer.transform(X_test_cleaned)

    print(f"TF-IDF vectorization complete. Training data shape: {X_train_full_vec.shape}, Test data shape: {X_test_vec.shape}")



    np.savez_compressed(os.path.join(output_dir, 'X_train_full_vec.npz'), data=X_train_full_vec.data, indices=X_train_full_vec.indices, indptr=X_train_full_vec.indptr, shape=X_train_full_vec.shape)
    np.save(os.path.join(output_dir, 'y_train_full.npy'), y_train_full)
    np.save(os.path.join(output_dir, 'train_sample_ids.npy'), train_sample_ids) # Save original IDs

    np.savez_compressed(os.path.join(output_dir, 'X_test_vec.npz'), data=X_test_vec.data, indices=X_test_vec.indices, indptr=X_test_vec.indptr, shape=X_test_vec.shape)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

  
    joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))

    print("Preprocessed data saved successfully.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)