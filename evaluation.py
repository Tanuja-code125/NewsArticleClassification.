import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from config import RESULTS_DIR
from models import build_pipeline

def train_and_evaluate(classifiers: dict, X_train, X_test, y_train, y_test, class_names: list):
    print(f"\n{Fore.CYAN}[STEP 4/4] Training and evaluating models...")

    results = []
    trained_models = {}
    model_loop = tqdm(classifiers.items(), desc=f"{Fore.CYAN}Overall Model Progress", unit="model")

    for name, classifier in model_loop:
        model_loop.set_postfix_str(f"Running: {name}")

        tqdm.write(f"\n{Fore.BLUE}{Style.BRIGHT}" + "=" * 55)
        tqdm.write(f"{Fore.BLUE}{Style.BRIGHT}Training & Evaluating: {Fore.YELLOW}{name}")
        tqdm.write(f"{Fore.BLUE}{Style.BRIGHT}" + "=" * 55)

        model = build_pipeline(classifier)
        note = " (RBF SVM/XGBoost may take a few minutes...)" if name in ["SVM RBF", "XGBoost"] else ""
        tqdm.write(f"{Fore.LIGHTBLACK_EX}  -> Vectorizing text and fitting {name}{note}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        tqdm.write(f"{Fore.GREEN}  -> Done in {training_time:.2f}s! Evaluating predictions...")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "Recall (%)": round(recall * 100, 2),
            "F1 Score (%)": round(f1 * 100, 2),
            "Training Time (seconds)": round(training_time, 3)
        })
        trained_models[name] = model

        tqdm.write(f"  Accuracy:  {Fore.GREEN}{Style.BRIGHT}{accuracy * 100:.2f}%")
        tqdm.write(f"  Precision: {Fore.CYAN}{precision * 100:.2f}%")
        tqdm.write(f"  Recall:    {Fore.CYAN}{recall * 100:.2f}%")
        tqdm.write(f"  F1 Score:  {Fore.GREEN}{Style.BRIGHT}{f1 * 100:.2f}%")
        tqdm.write(f"\n{Fore.MAGENTA}Classification Report:")
        tqdm.write(classification_report(
            y_test, predictions, target_names=class_names, zero_division=0
        ))

        _save_confusion_matrix(name, y_test, predictions, class_names)

    results_df = pd.DataFrame(results).sort_values(by="F1 Score (%)", ascending=False).reset_index(drop=True)
    return results_df, trained_models

def _save_confusion_matrix(model_name: str, y_test, predictions, class_names: list):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=class_names,
        xticks_rotation="vertical",
        ax=ax
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    fig.savefig(
        os.path.join(RESULTS_DIR, f"{filename}_confusion_matrix.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

def save_artifacts(results_df: pd.DataFrame, trained_models: dict, meta_info: dict) -> str:
    print(f"\n{Fore.BLUE}{Style.BRIGHT}========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    print(f"\n{Fore.CYAN}[INFO] Generating comparison graph and saving artifacts...")
    results_df.set_index("Model")[
        ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    ].plot(kind="bar", figsize=(12, 6))

    plt.title("News Classification Model Comparison")
    plt.ylabel("Score (%)")
    plt.xlabel("Machine Learning Model")
    plt.ylim(0, 105)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    best_name = results_df.iloc["Model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, os.path.join(RESULTS_DIR, "best_news_model.joblib"))

    meta_info["best_model"] = best_name
    joblib.dump(meta_info, os.path.join(RESULTS_DIR, "model_metadata.joblib"))

    return best_name
