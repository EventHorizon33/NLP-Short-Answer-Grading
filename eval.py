import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (safe for all environments)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─────────────────────────────────────────────
# CONFIG — update paths if needed
# ─────────────────────────────────────────────

TRAIN_PATH = r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-train.csv"
DEV_PATH   = r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-dev.csv"
TEST_PATH  = r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-test.csv"
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN  (self-contained, no dependency on preprocess.py)
# ─────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

def get_col(df, candidates):
    cols = df.columns.tolist()
    return next((c for c in candidates if c in cols), cols[0])

def validate_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Path is not a file: {path}")

def load_and_clean(path):
    validate_paths(path)
    df = pd.read_csv(path, sep='\t')
    text_col  = get_col(df, ['text','Text','tweet','sentence','body','answer'])
    label_col = get_col(df, ['label','Label','class','sentiment','answer_class'])
    df = df.dropna(subset=[text_col, label_col])
    df['clean_text'] = df[text_col].apply(clean_text)
    df = df[df['clean_text'].str.strip() != '']
    return df, text_col, label_col

print("Loading data...")
train_df, text_col, label_col = load_and_clean(TRAIN_PATH)
dev_df,   _,        _         = load_and_clean(DEV_PATH)
test_df,  _,        _         = load_and_clean(TEST_PATH)

# ─────────────────────────────────────────────
# 2. ENCODE LABELS
# ─────────────────────────────────────────────

le = LabelEncoder()
le.fit(train_df[label_col].astype(str))

y_train = le.transform(train_df[label_col].astype(str))
y_dev   = le.transform(dev_df[label_col].astype(str))
y_test  = le.transform(test_df[label_col].astype(str))

class_names = le.classes_.tolist()
print(f"Classes: {class_names}")

# ─────────────────────────────────────────────
# 3. TF-IDF VECTORIZE
# ─────────────────────────────────────────────

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_dev   = vectorizer.transform(dev_df['clean_text'])
X_test  = vectorizer.transform(test_df['clean_text'])

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'LinearSVC':           LinearSVC(max_iter=2000, C=1.0, random_state=42),
}

trained = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained[name] = model

# ─────────────────────────────────────────────
# 5. EVALUATION FUNCTION
# ─────────────────────────────────────────────

def evaluate(model, X, y_true, split_name, model_name, class_names):
    y_pred = model.predict(X)

    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted',  zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {model_name} — {split_name}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision (wtd)   : {prec:.4f}")
    print(f"  Recall    (wtd)   : {rec:.4f}")
    print(f"  F1 (weighted)     : {f1_w:.4f}")
    print(f"  F1 (macro)        : {f1_m:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                zero_division=0))

    return y_pred, {
        'model': model_name, 'split': split_name,
        'accuracy': acc, 'f1_weighted': f1_w,
        'f1_macro': f1_m, 'precision': prec, 'recall': rec
    }

# ─────────────────────────────────────────────
# 6. RUN EVALUATION ON DEV + TEST
# ─────────────────────────────────────────────

all_results = []
best_f1     = -1
best_model_name = None
best_preds  = None

for name, model in trained.items():
    # Dev set
    _, dev_metrics = evaluate(model, X_dev, y_dev, 'Dev', name, class_names)
    all_results.append(dev_metrics)

    # Test set
    y_pred_test, test_metrics = evaluate(model, X_test, y_test, 'Test', name, class_names)
    all_results.append(test_metrics)

    if test_metrics['f1_weighted'] > best_f1:
        best_f1         = test_metrics['f1_weighted']
        best_model_name = name
        best_preds      = y_pred_test

# ─────────────────────────────────────────────
# 7. CONFUSION MATRIX PLOT (best model on test)
# ─────────────────────────────────────────────

print(f"\nPlotting confusion matrix for best model: {best_model_name}")

cm = confusion_matrix(y_test, best_preds)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_model_name} (Test Set)', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved: {cm_path}")

# ─────────────────────────────────────────────
# 8. SUMMARY TABLE
# ─────────────────────────────────────────────

results_df = pd.DataFrame(all_results)
print("\n=== SUMMARY TABLE ===")
print(results_df[['model','split','accuracy','f1_weighted','f1_macro','precision','recall']]
      .to_string(index=False))

summary_path = os.path.join(OUTPUT_DIR, 'results_summary.csv')
results_df.to_csv(summary_path, index=False)
print(f"\nSaved summary: {summary_path}")

# ─────────────────────────────────────────────
# 9. ERROR ANALYSIS — top misclassified examples
# ─────────────────────────────────────────────

print("\n=== ERROR ANALYSIS (first 10 misclassified on test) ===")

best_model   = trained[best_model_name]
y_pred_test  = best_model.predict(X_test)
test_df_copy = test_df.copy().reset_index(drop=True)

errors = test_df_copy[y_pred_test != y_test].copy()
errors['predicted'] = le.inverse_transform(y_pred_test[y_pred_test != y_test])
errors['actual']    = le.inverse_transform(y_test[y_pred_test != y_test])

print(errors[[text_col, 'actual', 'predicted']].head(10).to_string(index=False))

errors_path = os.path.join(OUTPUT_DIR, 'errors.csv')
errors[[text_col, 'actual', 'predicted']].to_csv(errors_path, index=False)
print(f"\nSaved all errors: {errors_path}")

# ─────────────────────────────────────────────
# 10. SAVE BEST MODEL
# ─────────────────────────────────────────────

artifacts_dir = 'artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

with open(f'{artifacts_dir}/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open(f'{artifacts_dir}/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open(f'{artifacts_dir}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"\n✓ Best model ({best_model_name}, F1={best_f1:.4f}) saved to artifacts/")
print("✓ All done!")