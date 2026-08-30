"""
Comprehensive Machine Learning Training, Optimization, and Model Selection Pipeline.
Includes:
- Stratified 80/20 Train-Test splitting
- Text preprocessing & TF-IDF Vectorization
- Baseline model training: Multinomial Naive Bayes, Logistic Regression, Linear SVM
- Hyperparameter tuning using 3-Fold Stratified GridSearchCV
- Comparative evaluation & metrics export
- Automatic best model selection based on Test F1 Score
- Artifact serialization (models/, results/)
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Add src and parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess_corpus, preprocess_text
from src.evaluate import compute_metrics, plot_confusion_matrices, plot_model_comparison
from data.download_or_prepare import download_dataset, clean_and_prepare_data, OUTPUT_CSV

DATA_PATH = OUTPUT_CSV
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_or_prepare_dataset():
    """Load dataset, preparing it if not yet present."""
    if not os.path.exists(DATA_PATH):
        raw_path = download_dataset()
        df = clean_and_prepare_data(raw_path)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def run_training_pipeline(tune_hyperparameters: bool = True):
    """
    Main training execution function.
    """
    ensure_dirs()
    print("=" * 65)
    print("       AI FAKE NEWS DETECTOR - MODEL TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Load Data
    df = load_or_prepare_dataset()
    print(f"[1/7] Loaded dataset: {len(df)} total articles.")
    
    # Extract text and labels
    X_raw = df["combined_text"].values
    y_raw = df["label"].values
    
    # Label Encoding (FAKE -> 0, REAL -> 1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    print(f"      Classes mapped: {dict(zip(range(len(class_names)), class_names))}")
    
    # 2. Stratified 80/20 Train-Test Split (Reproducible random_state=42)
    print("\n[2/7] Splitting data into 80% Training and 20% Testing (Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training set: {len(X_train_raw)} samples")
    print(f"      Testing set:  {len(X_test_raw)} samples")
    
    # 3. NLP Preprocessing
    print("\n[3/7] Running NLP Preprocessing (cleaning, tokenization, stopwords, lemmatization)...")
    t0 = time.time()
    X_train_clean = preprocess_corpus(X_train_raw, verbose=False)
    X_test_clean = preprocess_corpus(X_test_raw, verbose=False)
    print(f"      Preprocessing completed in {time.time() - t0:.2f}s")
    
    # 4. Feature Extraction: TF-IDF Vectorization Experiments
    print("\n[4/7] Performing TF-IDF Feature Extraction & Experimentation...")
    # Experimenting with Unigrams + Bigrams with sublinear TF scaling
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # CRITICAL: Fit ONLY on training data to prevent data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    
    print(f"      TF-IDF Vocabulary Size: {X_train_tfidf.shape[1]} features")
    print(f"      Training feature matrix shape: {X_train_tfidf.shape}")
    print(f"      Testing feature matrix shape:  {X_test_tfidf.shape}")
    
    # 5. Model Training & Optimization
    print("\n[5/7] Training and Optimizing Classification Models...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    models_to_train = {}
    
    if tune_hyperparameters:
        print("      - Performing GridSearchCV Hyperparameter Tuning on Training Data...")
        
        # 5.1 Multinomial Naive Bayes
        print("        [+] Tuning Multinomial Naive Bayes...")
        nb_param_grid = {"alpha": [0.01, 0.1, 0.5, 1.0]}
        nb_grid = GridSearchCV(MultinomialNB(), nb_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        nb_grid.fit(X_train_tfidf, y_train)
        best_nb = nb_grid.best_estimator_
        print(f"            Best Naive Bayes Params: {nb_grid.best_params_} (CV F1: {nb_grid.best_score_:.4f})")
        models_to_train["Naive Bayes"] = best_nb
        
        # 5.2 Logistic Regression
        print("        [+] Tuning Logistic Regression...")
        lr_param_grid = {"C": [0.1, 1.0, 10.0, 50.0], "solver": ["lbfgs"], "max_iter": [1000]}
        lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        lr_grid.fit(X_train_tfidf, y_train)
        best_lr = lr_grid.best_estimator_
        print(f"            Best Logistic Regression Params: {lr_grid.best_params_} (CV F1: {lr_grid.best_score_:.4f})")
        models_to_train["Logistic Regression"] = best_lr
        
        # 5.3 Linear SVM (Calibrated for probability estimates)
        print("        [+] Tuning Linear SVM...")
        svm_param_grid = {"C": [0.1, 0.5, 1.0, 5.0]}
        svm_grid = GridSearchCV(LinearSVC(random_state=42, dual="auto", max_iter=2000), svm_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        svm_grid.fit(X_train_tfidf, y_train)
        best_svm_base = svm_grid.best_estimator_
        print(f"            Best Linear SVM Params: {svm_grid.best_params_} (CV F1: {svm_grid.best_score_:.4f})")
        
        # Calibrate SVM using CalibratedClassifierCV to get true probabilities
        calibrated_svm = CalibratedClassifierCV(best_svm_base, cv=3)
        calibrated_svm.fit(X_train_tfidf, y_train)
        models_to_train["Linear SVM"] = calibrated_svm
        
    else:
        # Fast baseline training
        nb = MultinomialNB(alpha=0.1)
        nb.fit(X_train_tfidf, y_train)
        models_to_train["Naive Bayes"] = nb
        
        lr = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        lr.fit(X_train_tfidf, y_train)
        models_to_train["Logistic Regression"] = lr
        
        base_svm = LinearSVC(C=1.0, random_state=42, dual="auto")
        calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)
        calibrated_svm.fit(X_train_tfidf, y_train)
        models_to_train["Linear SVM"] = calibrated_svm

    # 6. Evaluation on Unseen Test Split
    print("\n[6/7] Evaluating All Models on Independent Test Set (20%)...")
    metrics_list = []
    test_predictions = {}
    
    for name, model in models_to_train.items():
        y_pred = model.predict(X_test_tfidf)
        test_predictions[name] = y_pred
        metrics = compute_metrics(y_test, y_pred, model_name=name)
        metrics_list.append(metrics)
        print(f"      * {name:20s} -> Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | F1: {metrics['F1 Score']:.4f}")

    metrics_df = pd.DataFrame(metrics_list)
    
    # Save CSV and Plots
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"      Saved model metrics table: {csv_path}")
    
    comp_plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plot_model_comparison(metrics_df, comp_plot_path)
    
    cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plot_confusion_matrices(y_test, test_predictions, class_names, cm_plot_path)

    # 7. Model Selection & Serialization
    print("\n[7/7] Automatic Model Selection & Artifact Serialization...")
    # Select best model based on highest F1-Score
    best_row = metrics_df.sort_values(by="F1 Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_f1 = best_row["F1 Score"]
    best_model_obj = models_to_train[best_model_name]
    
    print(f"\n=======================================================")
    print(f"   WINNER / BEST MODEL: {best_model_name}")
    print(f"   Test Accuracy: {best_row['Accuracy']*100:.2f}% | Test F1 Score: {best_f1:.4f}")
    print(f"=======================================================\n")
    
    # Save best model, vectorizer, and encoder
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    
    # Also save all models for interactive model-switching in experiments
    for name, mdl in models_to_train.items():
        fname = f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(mdl, os.path.join(MODELS_DIR, fname))

    # Save model metadata
    metadata = {
        "best_model_name": best_model_name,
        "best_metrics": best_row.to_dict(),
        "classes": class_names,
        "features_count": int(X_train_tfidf.shape[1]),
        "training_samples": int(len(X_train_raw)),
        "testing_samples": int(len(X_test_raw)),
        "all_metrics": metrics_df.to_dict(orient="records"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SUCCESS] All model artifacts and evaluation graphs saved successfully.")
    return metadata


if __name__ == "__main__":
    run_training_pipeline(tune_hyperparameters=True)
