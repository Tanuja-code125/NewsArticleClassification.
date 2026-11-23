# 🗞️ Story Sorter — Multiclass News Article Classification Using NLP

An end-to-end Natural Language Processing (NLP) and Machine Learning system that automatically categorizes news articles and headlines into ten editorial desks. The project features a comparative benchmark of five supervised learning models trained on over 200,000 publications from the HuffPost news corpus, coupled with an editorial dark-themed interactive **Streamlit web application** (`app.py`) for live classification, batch prediction, and model interpretability.

---

## 🌟 Key Highlights & Features

- **High-Performance Models**: Champion **Multinomial Logistic Regression** model achieving **77.95% Accuracy** and **77.62% Weighted F1-Score**, with **Linear SVM** offering an ultra-fast **14-second** training alternative.
- **Top 10 Editorial Desks**: Categorizes stories into *Politics, Health & Wellness, Entertainment, Travel, Style & Beauty, Parenting, Queer Voices, Food & Drink, Business, and World News*.
- **Sublinear TF-IDF Feature Extraction**: Unigrams and bigrams (`ngram_range=(1, 2)`) across a 20,000-dimensional vocabulary with sublinear term-frequency scaling.
- **Editorial Dark Web App (`app.py`)**:
  - **📰 Live Classifier**: Real-time headline and body prediction, animated confidence ring gauge, probability distribution across all 10 desks, **close-call detection** (alerting when top 2 categories are within 8%), **feature attribution tokens** (`+weight`), and **in-text evidence highlighting** (`<mark>`).
  - **📂 Batch CSV Prediction**: Bulk file processing with multi-column text merging, persistent state, interactive **Altair category distribution charts**, preview table with progress-bar confidence columns, category filters, and 1-click export of predictions.
  - **📊 Model Benchmarks**: Champion KPI cards, interactive leaderboard with progress columns, grouped Altair comparison charts, and an interactive viewer for all model confusion matrices.
  - **ℹ️ Methodology & Taxonomy**: Step-by-step pipeline cards (*Clean $\rightarrow$ Vectorize $\rightarrow$ Classify*), color-coded desk taxonomy guide, and usage limitations.

---

## 🏗️ System Architecture

```
Raw News Text (Headline + Short Description)
                     │
                     ▼
       [1. Text Preprocessing]
         • Lowercase standardization
         • URL & HTML markup removal (<[^>]+>)
         • Non-alphabetic filtering ([a-z])
         • Whitespace compaction & deduplication
                     │
                     ▼
       [2. TF-IDF Feature Extraction]
         • Unigrams & Bigrams (ngram_range = 1, 2)
         • English stop words removal
         • Sublinear term frequency (1 + log(tf))
         • 20,000 maximum vocabulary features
         • L2 document vector normalization
                     │
                     ▼
       [3. Model Training & Evaluation]
         • Five Supervised Classifiers Benchmark
         • Champion: Multinomial Logistic Regression
                     │
                     ▼
       [4. Deployment & Interfaces]
         • CLI Interactive & Batch Predictor (predict.py)
         • Streamlit Editorial Web Application (app.py)
```

---

## 📊 Experimental Results & Benchmark

All five supervised machine learning models were trained on the identical 80% stratified training partition and evaluated against the 20% held-out test split:

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | Training Time (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **77.95** | **78.47** | **77.95** | **77.62** | 42.78 | 🏆 **Champion** |
| **Linear SVM** | 77.71 | 77.67 | 77.71 | 77.53 | **14.08** | ⚡ **Fastest Linear** |
| **SVM RBF** | 74.41 | 75.79 | 74.41 | 73.90 | 289.11 | Non-linear Kernel |
| **Multinomial Naive Bayes** | 74.12 | 76.31 | 74.12 | 72.78 | **7.15** | Ultra-Fast Baseline |
| **XGBoost Classifier** | 71.31 | 73.29 | 71.31 | 71.23 | 78.27 | Tree Ensemble |

### Key Experimental Insights:
1. **Linear Model Dominance**: High-dimensional sparse TF-IDF spaces (20,000 features) are inherently linearly separable. Regularized linear classifiers (Logistic Regression and Linear SVM) substantially outperformed non-linear kernels and tree ensembles.
2. **Linear SVM Efficiency**: Linear SVM achieved nearly identical performance to Logistic Regression with a **3x speedup** in training time (14.08s vs. 42.78s).
3. **XGBoost on Sparse Text**: Decision trees struggle on bag-of-words representations because axis-aligned orthogonal splits cannot easily capture dispersed linear combinations of vocabulary tokens.

---

## 🗂️ Project Structure

```
NewsArticleClassification/
├── app.py                     # Streamlit web application (Story Sorter)
├── config.py                  # Global configurations, file paths, and category mappings
├── evaluation.py              # Training loop, metric calculations, and artifact persistence
├── main.py                    # End-to-end training and evaluation execution pipeline
├── models.py                  # Scikit-learn and XGBoost model definitions & pipelines
├── predict.py                 # CLI predictor supporting interactive and batch CSV modes
├── preprocessing.py           # Text cleaning, category harmonization, and data splitting
├── requirements.txt           # Project dependencies
├── dataset/
│   └── News.csv               # Kaggle HuffPost News Category Dataset (~200k records)
└── results/
    ├── best_news_model.joblib # Serialized champion pipeline (TF-IDF + Logistic Regression)
    ├── model_metadata.joblib  # Label encoder, category list, and configuration metadata
    ├── model_comparison.csv   # Benchmark performance metrics table
    ├── model_comparison.png   # Comparative bar chart visualization
    ├── linear_svm_confusion_matrix.png
    ├── logistic_regression_confusion_matrix.png
    ├── naive_bayes_confusion_matrix.png
    ├── svm_rbf_confusion_matrix.png
    └── xgboost_confusion_matrix.png
```

---

## 💻 How to Run

### 1. Environment Setup & Installation
Clone the repository and install the required dependencies:

```bash
cd NewsArticleClassification
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train and Evaluate Models
To run the full preprocessing, training, and benchmarking pipeline across all five algorithms:

```bash
python main.py
```
*Trained artifacts, model comparison CSV, and confusion matrix visualizations will be saved to the `results/` folder.*

### 3. CLI Prediction
To run predictions directly from the command line:

- **Interactive Shell Mode**:
  ```bash
  python predict.py
  ```
- **Batch CSV Mode**:
  ```bash
  python predict.py path/to/articles.csv
  ```

### 4. Launch the Web Frontend (Story Sorter)
Start the modern interactive Streamlit web dashboard:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access:
- **Live Classifier**: Test custom articles, review confidence scores, inspect influential keywords, and view in-text highlighting.
- **Batch CSV Classifier**: Upload CSVs, merge headline/body columns, explore Altair category distributions, and download classified data.
- **Model Benchmarks**: View model leaderboards and inspect high-resolution confusion matrices.
- **Methodology & Taxonomy**: Explore the NLP pipeline and editorial category guide.

---

## 🏷️ Category Taxonomy (Top 10 Desks)

- 💼 **Business**: Markets, finance, economy, corporate earnings, and trade.
- 🎬 **Entertainment**: Cinema, television, music, celebrities, and pop culture.
- 🍲 **Food & Drink**: Culinary arts, recipes, restaurants, and gastronomy.
- 🌿 **Health & Wellness**: Medicine, fitness, nutrition, and mental health.
- 👶 **Parenting**: Childcare, family life, education, and parenting advice.
- 🏛️ **Politics**: Government, elections, legislation, and public policy.
- 🌈 **Queer Voices**: LGBTQ+ community, identity, rights, and culture.
- 👗 **Style & Beauty**: Fashion, design, cosmetics, apparel, and lifestyle.
- ✈️ **Travel**: Tourism, destinations, flights, hotels, and travel guides.
- 🌐 **World News**: International affairs, global conflicts, and diplomacy.

---

## 🛠️ Technologies Used

- **Python 3.13**
- **Scikit-learn**: TF-IDF Vectorization, Logistic Regression, Linear SVC, SVC (RBF), Multinomial Naive Bayes, LabelEncoder, Pipeline.
- **XGBoost**: Gradient boosted decision tree ensemble.
- **Streamlit**: Web application framework for live prediction, analytics, and file export.
- **Altair**: Interactive data visualization charts for category distribution and model metrics.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Matplotlib & Seaborn**: High-resolution confusion matrices and benchmark charts.
- **Joblib**: Model and metadata serialization.
- **Colorama & TQDM**: Styled CLI logging and training progress bars.
