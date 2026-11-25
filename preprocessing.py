# import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import (
    CATEGORY_OPTIONS,
    TEXT_OPTIONS,
    TITLE_OPTIONS,
    RANDOM_STATE,
    TEST_SIZE_RATIO,
    TOP_N_CATEGORIES,
    CATEGORY_MERGE_MAP
)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_prepare_data(dataset_path):
    path = Path(dataset_path)

    # 1. If path doesn't exist as given, try resolving relative to base directories
    if not path.exists():
        base_dir = Path(__file__).resolve().parent
        if (base_dir / path).exists():
            path = base_dir / path
        elif (base_dir.parent / path).exists():
            path = base_dir.parent / path

    # 2. If path is a directory, look for News.csv or the first .csv inside it
    if path.is_dir():
        news_file = path / "News.csv"
        if news_file.exists():
            path = news_file
        else:
            csv_candidates = list(path.glob("*.csv")) + list(path.glob("*.CSV"))
            if csv_candidates:
                path = csv_candidates[0]
            else:
                raise FileNotFoundError(
                    f"{Fore.RED}No CSV dataset found inside directory '{dataset_path}'.\n"
                    "Place News.csv inside the dataset folder."
                )

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"{Fore.RED}Dataset file not found at '{dataset_path}'.\n"
            "Place News.csv inside the dataset folder."
        )

    print(f"{Fore.CYAN}{Style.BRIGHT}[INFO] Checking and loading dataset from '{path}'...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    print(f"\n{Fore.BLUE}{Style.BRIGHT}========== DATASET INFORMATION ==========")
    print(f"Original rows: {Fore.YELLOW}{len(df)}")
    print(f"Columns:       {Fore.YELLOW}{df.columns.tolist()}")

    column_map = {str(col).strip().lower(): col for col in df.columns}

    # Detect target category column
    category_col = next(
        (column_map[name] for name in CATEGORY_OPTIONS if name in column_map),
        None
    )
    if category_col is None:
        raise ValueError(
            f"{Fore.RED}Could not detect the target category column.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Detect news text columns
    text_cols = [column_map[name] for name in TEXT_OPTIONS if name in column_map]
    title_cols = [column_map[name] for name in TITLE_OPTIONS if name in column_map]

    if not text_cols and not title_cols:
        raise ValueError(
            f"{Fore.RED}Could not detect news text columns.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    selected_cols = list(dict.fromkeys(title_cols + text_cols))
    print(f"Target column: {Fore.GREEN}{category_col}")
    print(f"Text columns:  {Fore.GREEN}{selected_cols}")

    # Combine title and article text (using tqdm directly to avoid pandas compatibility issues)
    print(f"\n{Fore.CYAN}[STEP 1/4] Combining title and article text...")
    data = df[selected_cols + [category_col]].copy()
    data = data.dropna(subset=[category_col])
    data["category"] = data[category_col].astype(str).str.strip().str.upper()

    # Merge redundant categories
    data["category"] = data["category"].replace(CATEGORY_MERGE_MAP)

    rows_iter = zip(*(data[col].tolist() for col in selected_cols))
    combined_texts = []
    for row_vals in tqdm(rows_iter, total=len(data), desc=f"{Fore.YELLOW}Combining text", bar_format="{l_bar}{bar:30}{r_bar}"):
        parts = []
        for value in row_vals:
            if pd.notna(value):
                s = str(value).strip()
                if s and s.lower() not in {"nan", "none", "null"}:
                    parts.append(s)
        combined_texts.append(" ".join(list(dict.fromkeys(parts))))

    data["news_text"] = combined_texts
    data = data[(data["news_text"].str.strip() != "") & (data["category"] != "")].copy()

    # Clean news text
    print(f"\n{Fore.CYAN}[STEP 2/4] Cleaning and normalizing text...")
    data["clean_text"] = [
        clean_text(text)
        for text in tqdm(data["news_text"].tolist(), desc=f"{Fore.YELLOW}Cleaning text ", bar_format="{l_bar}{bar:30}{r_bar}")
    ]
    data = data[data["clean_text"] != ""].copy()

    # Remove duplicates and conflicting labels
    print(f"\n{Fore.CYAN}[STEP 3/4] Removing duplicate and ambiguous articles...")
    before_duplicates = len(data)
    label_counts = data.groupby("clean_text")["category"].nunique()
    ambiguous_texts = label_counts[label_counts > 1].index

    data = data[~data["clean_text"].isin(ambiguous_texts)].copy()
    data = data.drop_duplicates(subset=["clean_text"], keep="first")

    # Keep only Top N categories
    if TOP_N_CATEGORIES is not None and data["category"].nunique() > TOP_N_CATEGORIES:
        top_categories = data["category"].value_counts().nlargest(TOP_N_CATEGORIES).index.tolist()
        before_filter = len(data)
        data = data[data["category"].isin(top_categories)].copy()
        print(f"\n{Fore.CYAN}[INFO] Filtered to Top {TOP_N_CATEGORIES} categories: {Fore.YELLOW}{top_categories}")
        print(f"Rows dropped from minor categories: {Fore.RED}{before_filter - len(data)}")

    print(f"\n{Fore.BLUE}{Style.BRIGHT}========== PREPROCESSING RESULTS ==========")
    print(f"Rows before duplicate removal: {Fore.YELLOW}{before_duplicates}")
    print(f"Rows after cleaning & filter:  {Fore.GREEN}{len(data)}")
    print(f"Unique articles:               {Fore.GREEN}{data['clean_text'].nunique()}")
    print(f"Number of categories:          {Fore.GREEN}{data['category'].nunique()}")
    print(f"\n{Fore.MAGENTA}Category distribution:")
    print(data["category"].value_counts())

    # Encode labels and split
    X = data["clean_text"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["category"])
    class_names = label_encoder.classes_.tolist()

    if len(class_names) < 2:
        raise ValueError(f"{Fore.RED}At least two different categories are required.")

    if pd.Series(y).value_counts().min() < 2:
        raise ValueError(f"{Fore.RED}At least one category has fewer than 2 articles.")

    test_count = max(int(round(len(data) * TEST_SIZE_RATIO)), len(class_names))
    if (len(data) - test_count) < len(class_names):
        raise ValueError(f"{Fore.RED}Not enough data for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n{Fore.BLUE}{Style.BRIGHT}========== TRAIN TEST SPLIT ==========")
    print(f"Total articles:    {Fore.YELLOW}{len(data)}")
    print(f"Training articles: {Fore.GREEN}{len(X_train)}")
    print(f"Testing articles:  {Fore.GREEN}{len(X_test)}")

    meta_info = {
        "target_column": category_col,
        "text_columns": selected_cols,
        "categories": class_names,
        "label_encoder": label_encoder
    }

    return X_train, X_test, y_train, y_test, meta_info
