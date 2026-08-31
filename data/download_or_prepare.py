"""
Data acquisition, multi-dataset consolidation, and exploratory analysis script.
Consolidates:
1. George McIntire Fake and Real News benchmark dataset (6,305 articles)
2. FakeNewsNet PolitiFact dataset (Politifact verified Real and Fake articles)
3. FakeNewsNet GossipCop dataset (GossipCop verified Real and Fake articles)
Generates a multi-domain, balanced dataset of 12,000+ clean news items with EDA plots.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(DATA_DIR, "news.csv")
RESULTS_DIR = os.path.join(os.path.dirname(DATA_DIR), "results")

# Benchmark source URLs
DATASET_SOURCES = {
    "mcintire": "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
    "fn_politifact_real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv",
    "fn_politifact_fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
    "fn_gossipcop_real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv",
    "fn_gossipcop_fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv",
}


def ensure_directories():
    """Ensure data and results directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def download_source(name: str, url: str) -> str:
    """Download individual CSV dataset source if not present."""
    local_path = os.path.join(DATA_DIR, f"raw_{name}.csv")
    if not os.path.exists(local_path):
        print(f"[INFO] Downloading {name} from {url} ...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"[INFO] Downloaded: {local_path}")
    return local_path


def load_and_standardize_datasets():
    """
    Download and standardize multi-source datasets:
    - McIntire dataset: title, text, label
    - FakeNewsNet PolitiFact & GossipCop: title, label
    """
    ensure_directories()
    all_dfs = []
    
    # 1. McIntire dataset
    mc_path = download_source("mcintire", DATASET_SOURCES["mcintire"])
    df_mc = pd.read_csv(mc_path)
    if "Unnamed: 0" in df_mc.columns:
        df_mc = df_mc.drop(columns=["Unnamed: 0"])
    df_mc = df_mc.rename(columns={"title": "title", "text": "text", "label": "label"})
    df_mc["label"] = df_mc["label"].astype(str).str.strip().str.upper()
    df_mc = df_mc[df_mc["label"].isin(["REAL", "FAKE"])].copy()
    df_mc["combined_text"] = df_mc["title"].fillna("") + " " + df_mc["text"].fillna("")
    df_mc["source"] = "McIntire"
    all_dfs.append(df_mc[["title", "text", "label", "combined_text", "source"]])
    print(f"[INFO] Loaded McIntire dataset: {len(df_mc)} records")
    
    # 2. FakeNewsNet PolitiFact (Real & Fake)
    try:
        p_real_path = download_source("fn_politifact_real", DATASET_SOURCES["fn_politifact_real"])
        p_fake_path = download_source("fn_politifact_fake", DATASET_SOURCES["fn_politifact_fake"])
        
        df_p_real = pd.read_csv(p_real_path)
        df_p_real["label"] = "REAL"
        df_p_real["text"] = df_p_real["title"]
        df_p_real["combined_text"] = df_p_real["title"]
        df_p_real["source"] = "FakeNewsNet_PolitiFact"
        all_dfs.append(df_p_real[["title", "text", "label", "combined_text", "source"]])
        
        df_p_fake = pd.read_csv(p_fake_path)
        df_p_fake["label"] = "FAKE"
        df_p_fake["text"] = df_p_fake["title"]
        df_p_fake["combined_text"] = df_p_fake["title"]
        df_p_fake["source"] = "FakeNewsNet_PolitiFact"
        all_dfs.append(df_p_fake[["title", "text", "label", "combined_text", "source"]])
        print(f"[INFO] Loaded FakeNewsNet PolitiFact: {len(df_p_real) + len(df_p_fake)} records")
    except Exception as e:
        print(f"[WARNING] Could not load FakeNewsNet PolitiFact: {e}")
        
    # 3. FakeNewsNet GossipCop (Balanced sample of Real & Fake)
    try:
        g_real_path = download_source("fn_gossipcop_real", DATASET_SOURCES["fn_gossipcop_real"])
        g_fake_path = download_source("fn_gossipcop_fake", DATASET_SOURCES["fn_gossipcop_fake"])
        
        df_g_real = pd.read_csv(g_real_path)
        df_g_fake = pd.read_csv(g_fake_path)
        
        # Take balanced sample (e.g. 3,000 real and 3,000 fake) to maintain equilibrium
        sample_n = min(len(df_g_fake), 3000)
        df_g_fake = df_g_fake.sample(n=sample_n, random_state=42)
        df_g_real = df_g_real.sample(n=sample_n, random_state=42)
        
        df_g_real["label"] = "REAL"
        df_g_real["text"] = df_g_real["title"]
        df_g_real["combined_text"] = df_g_real["title"]
        df_g_real["source"] = "FakeNewsNet_GossipCop"
        all_dfs.append(df_g_real[["title", "text", "label", "combined_text", "source"]])
        
        df_g_fake["label"] = "FAKE"
        df_g_fake["text"] = df_g_fake["title"]
        df_g_fake["combined_text"] = df_g_fake["title"]
        df_g_fake["source"] = "FakeNewsNet_GossipCop"
        all_dfs.append(df_g_fake[["title", "text", "label", "combined_text", "source"]])
        print(f"[INFO] Loaded FakeNewsNet GossipCop (balanced sample): {len(df_g_real) + len(df_g_fake)} records")
    except Exception as e:
        print(f"[WARNING] Could not load FakeNewsNet GossipCop: {e}")

    # Combine all
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean text
    merged_df["title"] = merged_df["title"].fillna("").astype(str).str.strip()
    merged_df["text"] = merged_df["text"].fillna("").astype(str).str.strip()
    merged_df["combined_text"] = merged_df["combined_text"].fillna("").astype(str).str.strip()
    
    # Filter short texts & remove duplicate combinations
    merged_df = merged_df[merged_df["combined_text"].str.len() >= 15].copy()
    merged_df = merged_df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    
    # Balance classes if slight difference
    real_count = (merged_df["label"] == "REAL").sum()
    fake_count = (merged_df["label"] == "FAKE").sum()
    min_count = min(real_count, fake_count)
    
    real_subset = merged_df[merged_df["label"] == "REAL"].sample(n=min_count, random_state=42)
    fake_subset = merged_df[merged_df["label"] == "FAKE"].sample(n=min_count, random_state=42)
    balanced_df = pd.concat([real_subset, fake_subset], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save combined dataset
    balanced_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[SUCCESS] Prepared expanded balanced dataset saved to: {OUTPUT_CSV} ({len(balanced_df)} articles)")
    return balanced_df


def perform_eda(df, save_plot=True):
    """Exploratory data analysis on the expanded dataset."""
    ensure_directories()
    print("\n" + "=" * 55)
    print("      EXPANDED DATASET EXPLORATORY ANALYSIS")
    print("=" * 55)
    
    class_counts = df["label"].value_counts()
    total_samples = len(df)
    
    print(f"Total Consolidated Articles: {total_samples}")
    print(f"REAL news count:             {class_counts.get('REAL', 0)} ({class_counts.get('REAL', 0)/total_samples*100:.2f}%)")
    print(f"FAKE news count:             {class_counts.get('FAKE', 0)} ({class_counts.get('FAKE', 0)/total_samples*100:.2f}%)")
    
    df["word_count"] = df["combined_text"].apply(lambda x: len(str(x).split()))
    print(f"Average Word Count:          {df['word_count'].mean():.1f} words (Median: {df['word_count'].median():.0f})")
    
    if "source" in df.columns:
        print("\nSamples by Source:")
        print(df["source"].value_counts())
    print("=" * 55 + "\n")
    
    if save_plot:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        palette = {"REAL": "#10b981", "FAKE": "#ef4444"}
        sns.countplot(data=df, x="label", hue="label", legend=False, palette=palette, ax=axes[0])
        axes[0].set_title("Expanded Dataset Class Balance", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel("Class Label")
        axes[0].set_ylabel("Articles Count")
        
        for p in axes[0].patches:
            h = p.get_height()
            axes[0].annotate(f"{int(h)}\n({h/total_samples*100:.1f}%)",
                            (p.get_x() + p.get_width() / 2., h / 2),
                            ha="center", va="center", fontsize=11, color="white", fontweight="bold")
                            
        p95 = df["word_count"].quantile(0.95)
        filtered_df = df[df["word_count"] <= p95]
        sns.histplot(data=filtered_df, x="word_count", hue="label", palette=palette, kde=True, element="step", bins=30, ax=axes[1])
        axes[1].set_title("Article Word Length Distribution", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_xlabel("Word Count")
        axes[1].set_ylabel("Density / Count")
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "eda_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[INFO] Updated EDA plot saved to: {plot_path}")


def main():
    df = load_and_standardize_datasets()
    perform_eda(df, save_plot=True)


if __name__ == "__main__":
    main()
