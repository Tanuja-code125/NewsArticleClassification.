import os
import sys
import joblib
import pandas as pd
from colorama import init, Fore, Style
from config import RESULTS_DIR
from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "best_news_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "results", "model_metadata.joblib")

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"{Fore.RED}Model artifacts not found in '{RESULTS_DIR}/'.\n"
            "Run 'python main.py' first to train and save the best model."
        )

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def predict_article(text: str, model, label_encoder) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        return {"category": "Unknown (empty after cleaning)", "confidence": None}

    # Predict numeric label and decode back to original category string
    pred_idx = model.predict([cleaned])
    category = label_encoder.inverse_transform(pred_idx)[0]

    # Check if the winning classifier supports probability estimates (e.g., NB, LR, XGBoost)
    confidence = None
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probs = model.predict_proba([cleaned])
        confidence = round(float(probs.max()) * 100, 2)

    return {"category": category, "confidence": confidence}


def interactive_mode(model, metadata):
    label_encoder = metadata["label_encoder"]
    best_model_name = metadata.get("best_model", "Unknown")

    print(f"\n{Fore.BLUE}{Style.BRIGHT}========== NEWS ARTICLE CLASSIFIER ==========")
    print(f"Loaded Model: {Fore.GREEN}{best_model_name}")
    print(f"Categories:   {Fore.YELLOW}{', '.join(metadata['categories'])}")
    print(f"{Fore.LIGHTBLACK_EX}Type or paste a headline/article below (type 'exit' or 'quit' to stop).\n")

    while True:
        try:
            user_input = input(f"{Fore.CYAN}Enter news text > {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.YELLOW}Exiting predictor.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print(f"{Fore.YELLOW}Exiting predictor.")
            break

        if not user_input:
            continue

        result = predict_article(user_input, model, label_encoder)
        conf_str = f" ({result['confidence']}% confidence)" if result["confidence"] is not None else ""
        print(f"  -> Predicted Category: {Fore.GREEN}{Style.BRIGHT}{result['category']}{Fore.CYAN}{conf_str}\n")


def batch_csv_mode(csv_path: str, model, metadata):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{Fore.RED}Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    text_cols = [c for c in metadata["text_columns"] if c in df.columns]

    if not text_cols:
        # Fallback to first text-like column if original column names differ
        text_cols = [df.columns]
        print(f"{Fore.YELLOW}[WARN] Original text columns not found; using column '{text_cols}'.")

    print(f"{Fore.CYAN}[INFO] Running batch predictions on {len(df)} rows using columns: {text_cols}...")

    combined = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    cleaned = combined.apply(clean_text)

    preds = model.predict(cleaned)
    df["predicted_category"] = metadata["label_encoder"].inverse_transform(preds)

    output_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"{Fore.GREEN}[DONE] Predictions saved to: {output_path}")


if __name__ == "__main__":
    init(autoreset=True)
    model, metadata = load_artifacts()

    # If a CSV path is passed as an argument, run batch mode; otherwise start interactive mode
    if len(sys.argv) > 1:
        batch_csv_mode(sys.argv, model, metadata)
    else:
        interactive_mode(model, metadata)
