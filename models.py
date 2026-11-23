from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from config import RANDOM_STATE, TFIDF_PARAMS

def get_classifiers() -> dict:
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
        "SVM RBF": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
    }

def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("classifier", classifier)
    ])
