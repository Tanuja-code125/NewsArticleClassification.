from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "News.csv"
RESULTS_DIR = BASE_DIR / "results"
RANDOM_STATE = 42
TEST_SIZE_RATIO = 0.20
TOP_N_CATEGORIES = 10

CATEGORY_OPTIONS = [
    "category", "news_category", "news category",
    "label", "labels", "class", "target", "subject", "type"
]

TEXT_OPTIONS = [
    "text", "full_text", "article_text", "content",
    "news", "article", "description", "short_description", "body"
]

TITLE_OPTIONS = ["title", "headline", "news_title"]

# Merge redundant HuffPost categories before picking Top 10
CATEGORY_MERGE_MAP = {
    "THE WORLDPOST": "WORLD NEWS",
    "WORLDPOST": "WORLD NEWS",
    "WELLNESS": "HEALTH & WELLNESS",
    "HEALTHY LIVING": "HEALTH & WELLNESS",
    "STYLE": "STYLE & BEAUTY",
    "ARTS & CULTURE": "ARTS",
    "CULTURE & ARTS": "ARTS",
    "PARENTS": "PARENTING",
    "GREEN": "ENVIRONMENT",
    "SCIENCE": "TECH & SCIENCE",
    "TECH": "TECH & SCIENCE"
}

TFIDF_PARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "sublinear_tf": True
}

def ensure_directories():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
