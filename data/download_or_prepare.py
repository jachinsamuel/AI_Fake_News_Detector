"""
Data acquisition, preparation, and exploratory analysis script.
Downloads and normalizes the Fake and Real News dataset, handles missing values,
generates combined text features, and produces EDA visualizations.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(DATA_DIR, "news.csv")
RESULTS_DIR = os.path.join(os.path.dirname(DATA_DIR), "results")


def ensure_directories():
    """Ensure data and results directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def download_dataset(target_path=OUTPUT_CSV):
    """Download the benchmark Fake or Real News dataset if not present."""
    ensure_directories()
    raw_path = os.path.join(DATA_DIR, "raw_news.csv")
    
    if not os.path.exists(raw_path):
        print(f"[INFO] Downloading dataset from {DATASET_URL} ...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(DATASET_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(raw_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"[INFO] Download completed: {raw_path}")
    else:
        print(f"[INFO] Raw dataset already exists at: {raw_path}")
        
    return raw_path


def clean_and_prepare_data(raw_csv_path, output_csv_path=OUTPUT_CSV):
    """
    Clean the dataset:
    - Standardize column names (title, text, label)
    - Normalize labels to uppercase 'REAL' and 'FAKE'
    - Handle missing values & remove duplicate entries
    - Create 'combined_text' = title + " " + text
    """
    print("[INFO] Processing and cleaning dataset...")
    df = pd.read_csv(raw_csv_path)
    
    # Handle unnamed ID columns if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ["title", "headline"]:
            col_mapping[col] = "title"
        elif c_lower in ["text", "article", "content"]:
            col_mapping[col] = "text"
        elif c_lower in ["label", "target", "class"]:
            col_mapping[col] = "label"
            
    df = df.rename(columns=col_mapping)
    
    # Verify required columns
    required_cols = ["title", "text", "label"]
    for req in required_cols:
        if req not in df.columns:
            raise ValueError(f"Missing required column '{req}' in dataset. Found: {list(df.columns)}")
            
    # Normalize labels: REAL vs FAKE
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["REAL", "FAKE"])].copy()
    
    # Fill missing titles/texts with empty strings
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    
    # Create combined text
    df["combined_text"] = df["title"] + " " + df["text"]
    df["combined_text"] = df["combined_text"].str.strip()
    
    # Drop rows with negligible content (less than 15 characters)
    initial_len = len(df)
    df = df[df["combined_text"].str.len() >= 15].copy()
    
    # Remove duplicate records
    df = df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    final_len = len(df)
    
    print(f"[INFO] Raw records: {initial_len} -> Clean unique records: {final_len}")
    
    # Save cleaned dataset
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Prepared dataset saved to: {output_csv_path}")
    return df


def perform_eda(df, save_plot=True):
    """
    Perform Exploratory Data Analysis:
    - Count real vs fake
    - Class distribution percentages
    - Character and word length statistics
    - Generate summary plots
    """
    ensure_directories()
    print("\n" + "=" * 50)
    print("      EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)
    
    class_counts = df["label"].value_counts()
    total_samples = len(df)
    
    print(f"Total Clean Samples: {total_samples}")
    print(f"REAL news count:     {class_counts.get('REAL', 0)} ({class_counts.get('REAL', 0)/total_samples*100:.2f}%)")
    print(f"FAKE news count:     {class_counts.get('FAKE', 0)} ({class_counts.get('FAKE', 0)/total_samples*100:.2f}%)")
    
    # Word count and character count analysis
    df["word_count"] = df["combined_text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["combined_text"].apply(lambda x: len(str(x)))
    
    print(f"Average Word Count:  {df['word_count'].mean():.1f} words (Median: {df['word_count'].median():.0f})")
    print(f"Average Char Count:  {df['char_count'].mean():.1f} chars")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("=" * 50 + "\n")
    
    if save_plot:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Class distribution bar chart
        palette = {"REAL": "#10b981", "FAKE": "#ef4444"}
        sns.countplot(
            data=df, 
            x="label", 
            hue="label",
            legend=False,
            palette=palette, 
            ax=axes[0]
        )
        axes[0].set_title("Dataset Class Distribution (REAL vs FAKE)", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel("Class Label", fontsize=11)
        axes[0].set_ylabel("Number of Articles", fontsize=11)
        for p in axes[0].patches:
            height = p.get_height()
            axes[0].annotate(f'{int(height)}\n({height/total_samples*100:.1f}%)',
                            (p.get_x() + p.get_width() / 2., height / 2),
                            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
                            
        # 2. Word count distribution by class (capped at 95th percentile for clean display)
        p95 = df["word_count"].quantile(0.95)
        filtered_df = df[df["word_count"] <= p95]
        sns.histplot(
            data=filtered_df,
            x="word_count",
            hue="label",
            palette=palette,
            kde=True,
            element="step",
            bins=35,
            ax=axes[1]
        )
        axes[1].set_title("Article Word Length Distribution by Class", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_xlabel("Word Count (up to 95th percentile)", fontsize=11)
        axes[1].set_ylabel("Density / Count", fontsize=11)
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "eda_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[INFO] EDA Visualization saved to: {plot_path}")


def main():
    raw_path = download_dataset()
    df = clean_and_prepare_data(raw_path)
    perform_eda(df, save_plot=True)


if __name__ == "__main__":
    main()
