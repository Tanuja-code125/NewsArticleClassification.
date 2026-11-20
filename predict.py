
import joblib

model_path = "results/best_news_model.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print("Model not found. Please run main.py first.")
    raise SystemExit(1)

print("\nNEWS CATEGORY PREDICTION")
print("Type a news article below.")
print("Type 'exit' to stop.")

while True:
    news = input("\nEnter news article: ").strip()

    if news.lower() == "exit":
        print("Thank you!")
        break

    if not news:
        print("Please enter some news text.")
        continue

    category = model.predict([news])[0]
    print("Predicted category:", category)
