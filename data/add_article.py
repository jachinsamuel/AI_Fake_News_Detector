"""
Helper Utility: Add New Articles to the Dataset & Retrain.
Allows adding single custom news samples or importing an external CSV,
automatically generating combined_text, updating data/news.csv, and optionally retraining models.
"""

import os
import sys
import argparse
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CSV_PATH = os.path.join(ROOT_DIR, "data", "news.csv")


def add_single_sample(title: str, text: str, label: str, source: str = "User_Contributed", retrain: bool = False):
    """Append a single verified article to data/news.csv."""
    clean_label = label.strip().upper()
    if clean_label not in ("REAL", "FAKE"):
        raise ValueError("Label must be either 'REAL' or 'FAKE'.")

    clean_title = title.strip()
    clean_text = text.strip()
    combined = f"{clean_title} {clean_text}".strip()

    new_row = pd.DataFrame([{
        "title": clean_title,
        "text": clean_text,
        "label": clean_label,
        "combined_text": combined,
        "source": source
    }])

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # Deduplicate by text
    df = df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Added 1 article. Total dataset size: {len(df)} samples.")

    if retrain:
        print("\n[INFO] Triggering model retraining pipeline...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT_DIR, "src", "train.py")], check=True)


def import_external_csv(file_path: str, title_col: str = "title", text_col: str = "text", label_col: str = "label", retrain: bool = False):
    """Import and merge an external dataset CSV into data/news.csv."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext_df = pd.read_csv(file_path)
    print(f"[INFO] Loaded external CSV with {len(ext_df)} rows.")

    required_cols = [title_col, text_col, label_col]
    for col in required_cols:
        if col not in ext_df.columns:
            raise ValueError(f"Column '{col}' not found in external CSV. Available columns: {list(ext_df.columns)}")

    formatted_df = pd.DataFrame({
        "title": ext_df[title_col].fillna(""),
        "text": ext_df[text_col].fillna(""),
        "label": ext_df[label_col].astype(str).str.strip().str.upper(),
        "source": f"Imported_{os.path.basename(file_path)}"
    })

    # Standardize labels (handle 1/0 or REAL/FAKE)
    label_map = {"1": "REAL", "0": "FAKE", "TRUE": "REAL", "FALSE": "FAKE"}
    formatted_df["label"] = formatted_df["label"].replace(label_map)
    formatted_df = formatted_df[formatted_df["label"].isin(["REAL", "FAKE"])]

    formatted_df["combined_text"] = formatted_df["title"] + " " + formatted_df["text"]

    if os.path.exists(CSV_PATH):
        main_df = pd.read_csv(CSV_PATH)
        merged_df = pd.concat([main_df, formatted_df], ignore_index=True)
    else:
        merged_df = formatted_df

    merged_df = merged_df.drop_duplicates(subset=["combined_text"]).dropna(subset=["combined_text"])
    merged_df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Successfully imported data! New total dataset size: {len(merged_df)} samples.")

    if retrain:
        print("\n[INFO] Triggering model retraining pipeline...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT_DIR, "src", "train.py")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add news data to dataset and retrain model.")
    parser.add_argument("--title", type=str, help="News article title / headline")
    parser.add_argument("--text", type=str, help="News article body text")
    parser.add_argument("--label", type=str, choices=["REAL", "FAKE", "real", "fake"], help="Ground-truth label (REAL or FAKE)")
    parser.add_argument("--file", type=str, help="Path to external CSV file to import")
    parser.add_argument("--retrain", action="store_true", help="Automatically retrain the model after adding data")

    args = parser.parse_args()

    if args.file:
        import_external_csv(args.file, retrain=args.retrain)
    elif args.title and args.text and args.label:
        add_single_sample(args.title, args.text, args.label, retrain=args.retrain)
    else:
        print("\n" + "=" * 60)
        print("  AI FAKE NEWS DETECTOR - INTERACTIVE DATA CONTRIBUTOR")
        print("=" * 60)
        t = input("Enter Article Title / Headline: ").strip()
        txt = input("Enter Article Body Text: ").strip()
        lbl = input("Enter Label (REAL or FAKE): ").strip().upper()
        ret = input("Do you want to retrain the model now? (y/n): ").strip().lower() == "y"
        if t and txt and lbl in ("REAL", "FAKE"):
            add_single_sample(t, txt, lbl, retrain=ret)
        else:
            print("[ERROR] Title, text, and valid label (REAL/FAKE) are required.")
