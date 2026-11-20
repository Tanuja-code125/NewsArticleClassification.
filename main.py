
import os
import re
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results", exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

file_path = "dataset/news_classification_dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Dataset not found! Put your CSV file inside the dataset folder."
    )

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==========================================
# 3. FIND CATEGORY COLUMN
# ==========================================

category_options = [
    "Category", "category",
    "Label", "label",
    "Class", "class",
    "News_Category"
]

category_col = next(
    (col for col in category_options if col in df.columns),
    None
)

if category_col is None:
    raise ValueError("Could not find the category column.")

# ==========================================
# 4. COMBINE AVAILABLE TEXT COLUMNS
# ==========================================

text_options = [
    "Title", "Article_Text", "Full_Text", "Keywords",
    "title", "article_text", "full_text", "keywords",
    "Text", "text", "Content", "content"
]

text_columns = [
    col for col in text_options if col in df.columns
]

if not text_columns:
    raise ValueError("Could not find any news text columns.")

print("\nText columns used:", text_columns)

# Display unique values to help identify repeated text.
for col in text_columns:
    values = df[col].dropna().astype(str).str.strip()
    print(
        f"{col}: {values[values != ''].nunique()} unique values"
    )

# ==========================================
# 5. CLEAN AND PREPARE DATA
# ==========================================

data = df.copy()

data[category_col] = (
    data[category_col].fillna("").astype(str).str.strip()
)

data = data[data[category_col] != ""].copy()


def combine_text(row):
    parts = []

    for col in text_columns:
        value = row[col]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "nan":
                parts.append(value)

    # Remove repeated fields within the same row.
    parts = list(dict.fromkeys(parts))

    return " ".join(parts)


data["news_text"] = data.apply(combine_text, axis=1)

# ==========================================
# 6. TEXT PREPROCESSING
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["clean_text"] = data["news_text"].apply(clean_text)

data = data[data["clean_text"] != ""].copy()

# Do not remove rows just because their text repeats.
# This allows the pipeline to run, but repeated data can
# make evaluation scores unreliable.

print("\n========== CLEANING RESULTS ==========")
print("Rows available:", len(data))
print("Unique text examples:", data["clean_text"].nunique())
print("Categories:", data[category_col].nunique())
print("\nRows per category:")
print(data[category_col].value_counts())

if data["clean_text"].nunique() < 100:
    print(
        "\nWARNING: Very few unique text examples were found."
        "\nThe program can run, but scores may be unreliable."
        "\nCheck the dataset before using results in your report."
    )

# ==========================================
# 7. PREPARE INPUT AND OUTPUT
# ==========================================

X = data["clean_text"]
y = data[category_col]

if y.nunique() < 2:
    raise ValueError("At least two categories are required.")

counts = y.value_counts()

if counts.min() < 2:
    raise ValueError(
        "Every category must contain at least two rows."
    )

number_of_categories = y.nunique()

# The test set must be large enough to include every category.
test_count = max(
    number_of_categories,
    int(round(len(data) * 0.20))
)

if test_count >= len(data):
    raise ValueError("Not enough rows to split the dataset.")

# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_count,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# ==========================================
# 9. DEFINE THREE MODELS
# ==========================================

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}

results = []
trained_models = {}

# ==========================================
# 10. TRAIN AND EVALUATE MODELS
# ==========================================

for name, classifier in models.items():

    print("\n===================================")
    print("Training:", name)
    print("===================================")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (seconds)": training_time
    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Training time:", round(training_time, 3), "seconds")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save confusion matrix.
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=sorted(y.unique()),
        xticks_rotation="vertical",
        cmap="Blues"
    )

    display.ax_.set_title(name + " - Confusion Matrix")
    display.figure_.tight_layout()

    filename = name.lower().replace(" ", "_")

    display.figure_.savefig(
        f"results/{filename}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(display.figure_)

# ==========================================
# 11. SAVE MODEL COMPARISON CSV
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# ==========================================
# 12. SAVE COMPARISON GRAPH
# ==========================================

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("News Article Classification Model Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(
    best_model,
    "results/best_news_model.joblib"
)

# ==========================================
# 14. DISPLAY FINAL RESULTS
# ==========================================

print("\n========== PROJECT COMPLETED ==========")
print("Best model by weighted F1 Score:", best_name)

print("\nFiles saved in the results folder:")
print("1. model_comparison.csv")
print("2. model_comparison.png")
print("3. naive_bayes_confusion_matrix.png")
print("4. logistic_regression_confusion_matrix.png")
print("5. linear_svm_confusion_matrix.png")
print("6. best_news_model.joblib")

print("\nCheck the results folder in VS Code.")
