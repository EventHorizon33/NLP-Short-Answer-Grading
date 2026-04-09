# Answer Grading for Short Answers

Final project for the Natural Language Processing course.  
Automatically grades short answers from community Q&A forums using classical machine learning on the SemEval-2013 CQA dataset.

---

## Problem Statement

Given a short answer text, classify it as **good (1)** or **bad (0)** — i.e., determine whether the answer is a relevant, useful response to a question. Framed as binary text classification.

---

## Dataset

**SemEval-2013 Community Question Answering (Task 3)**  
Source: [Kaggle — SemEval-datasets](https://www.kaggle.com/datasets/azzouza2018/semevaldatadets)

| File | Purpose |
|------|---------|
| `semeval-2013-train.csv` | Model training |
| `semeval-2013-dev.csv` | Validation / hyperparameter tuning |
| `semeval-2013-test.csv` | Final evaluation |

Each CSV has two columns: `text` (the answer string) and `label` (0 or 1).

> **Note:** The dataset files are not included in this repository. Download them from Kaggle and place them in the project root before running.

---

## Project Structure

```
NLP-Project/
│
├── preproc.py               # Text cleaning + TF-IDF vectorization
├── train_and_evaluate.py    # Model training + full evaluation pipeline
│
├── results/
│   ├── confusion_matrix.png # Confusion matrix for best model
│   ├── results_summary.csv  # All metrics across models and splits
│   └── errors.csv           # Misclassified examples (error analysis)
│
├── artifacts/               # Saved model + vectorizer (generated on run)
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── NLP_Project_Report.docx  # Final project report
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn nltk matplotlib
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/azzouza2018/semevaldatadets) and place these three files in the project root:
- `semeval-2013-train.csv`
- `semeval-2013-dev.csv`
- `semeval-2013-test.csv`

---

## Usage

Run the scripts in order:

```bash
# Step 1 — Preprocess (optional standalone run to verify data loads correctly)
python preproc.py

# Step 2 — Train models and evaluate (runs preprocessing internally too)
python train_and_evaluate.py
```

Output will be saved to `results/` and `artifacts/`.

---

## Pipeline

```
Raw CSV  →  Clean Text  →  TF-IDF Vectors  →  Classifier  →  Evaluation
```

| Step | Details |
|------|---------|
| Preprocessing | Lowercase, remove URLs/mentions/hashtags, strip punctuation, remove stopwords, lemmatize |
| Vectorization | TF-IDF, top 10,000 features, unigrams + bigrams, sublinear TF scaling |
| Models | Logistic Regression, LinearSVC (both with C=1.0) |
| Evaluation | Accuracy, Precision, Recall, F1 (weighted + macro), Confusion Matrix, Error Analysis |

---

## Results

| Model | Split | Accuracy | F1 (weighted) |
|-------|-------|----------|----------------|
| Logistic Regression | Dev  | — | — |
| Logistic Regression | Test | — | — |
| LinearSVC           | Dev  | — | — |
| LinearSVC           | Test | — | — |

> Fill in your actual numbers from `results/results_summary.csv`

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| scikit-learn | ≥ 1.0 |
| pandas | ≥ 1.3 |
| nltk | ≥ 3.7 |
| matplotlib | ≥ 3.4 |

---

## References

1. Nakov et al. (2015). SemEval-2015 Task 3: Answer Selection in Community Question Answering. *Proceedings of SemEval-2015.*
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
3. Bird, Klein & Loper (2009). *Natural Language Processing with Python.* O'Reilly Media.