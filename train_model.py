import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

df = pd.read_csv("dataset/News.csv", on_bad_lines="skip")
df = df.dropna(subset=["category"])

headline = df["headline"].fillna("").str.replace(r"\s+authors$", "", regex=True)
df["text"] = headline + " " + df["short_description"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              min_df=3, max_features=100000, sublinear_tf=True)),
    ("clf", LinearSVC(C=0.5)),
])

print("Training on", len(X_train), "articles... (1-2 minutes)")
model.fit(X_train, y_train)

print("Test accuracy:", round(model.score(X_test, y_test), 3))
for t in [
    "The captain scored a late winner in extra time as the home side beat their rivals 2-1 and booked a place in the final.",
    "The Senate voted late Thursday to pass a bipartisan spending bill.",
]:
    print(model.predict([t])[0], "<-", t[:50])

joblib.dump(model, "model.pkl")
print("Model saved!")