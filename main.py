from colorama import init, Fore, Style
from config import DATASET_PATH, RESULTS_DIR, ensure_directories
from preprocessing import load_and_prepare_data
from models import get_classifiers
from evaluation import train_and_evaluate, save_artifacts

def main():
    init(autoreset=True)
    ensure_directories()

    # 1. Load, clean, and split data
    X_train, X_test, y_train, y_test, meta_info = load_and_prepare_data(DATASET_PATH)

    # 2. Initialize classifiers
    classifiers = get_classifiers()

    # 3. Train and evaluate all models
    results_df, trained_models = train_and_evaluate(
        classifiers, X_train, X_test, y_train, y_test, meta_info["categories"]
    )

    # 4. Save comparison reports, plots, and best model
    best_name = save_artifacts(results_df, trained_models, meta_info)

    # 5. Summary
    print(f"\n{Fore.GREEN}{Style.BRIGHT}========== PROJECT COMPLETED ==========")
    print(f"Dataset:          {Fore.YELLOW}{DATASET_PATH}")
    print(f"Best model (F1):  {Fore.GREEN}{Style.BRIGHT}{best_name}")
    print(f"Results folder:   {Fore.CYAN}{RESULTS_DIR}")
    print(f"Comparison table: {Fore.CYAN}results/model_comparison.csv")
    print(f"Comparison graph: {Fore.CYAN}results/model_comparison.png")
    print(f"Best model file:  {Fore.CYAN}results/best_news_model.joblib")
    print(f"Model metadata:   {Fore.CYAN}results/model_metadata.joblib")
    print(f"{Fore.GREEN}Confusion matrices saved for all five models.")

if __name__ == "__main__":
    main()
