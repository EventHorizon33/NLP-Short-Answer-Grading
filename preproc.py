import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def validate_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Path is not a file: {path}")

def load_data(train_path, dev_path, test_path):
    validate_paths(train_path, dev_path, test_path)

    train = pd.read_csv(train_path, sep='\t')
    dev   = pd.read_csv(dev_path, sep='\t')
    test  = pd.read_csv(test_path, sep='\t')

    print("=== Raw Data Shapes ===")
    print(f"Train : {train.shape}")
    print(f"Dev   : {dev.shape}")
    print(f"Test  : {test.shape}")

    print("\n=== Columns ===")
    print(train.columns.tolist())

    print("\n=== Sample rows ===")
    print(train.head(3))

    return train, dev, test


# ─────────────────────────────────────────────
# 2. IDENTIFY TEXT & LABEL COLUMNS
# ─────────────────────────────────────────────

def get_column_names(df):
    cols = df.columns.tolist()

    if len(cols) < 2:
        raise ValueError(f"Expected at least 2 columns, found {len(cols)}: {cols}")

    label_candidates = ['label', 'Label', 'class', 'sentiment', 'answer_class']
    text_candidates  = ['text', 'Text', 'tweet', 'sentence', 'body', 'answer']

    label_col = next((c for c in label_candidates if c in cols), None)
    text_col  = next((c for c in text_candidates  if c in cols), None)

    if label_col is None or text_col is None:
        print(f"Warning: could not auto-detect text/label columns. Falling back to first two columns: {cols[:2]}")
    label_col = label_col or cols[0]
    text_col  = text_col  or cols[1]

    print(f"\nUsing label column : '{label_col}'")
    print(f"Using text column  : '{text_col}'")
    return text_col, label_col


# ─────────────────────────────────────────────
# 3. CLEAN TEXT
# ─────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove @mentions and #hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]

    return ' '.join(tokens)


# ─────────────────────────────────────────────
# 4. PREPROCESS DATAFRAME
# ─────────────────────────────────────────────

def preprocess_df(df, text_col, label_col):
    df = df.copy()

    # Drop rows with missing text or label
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN values")

    # Clean text
    df['clean_text'] = df[text_col].apply(clean_text)

    # Drop rows where cleaning produced empty string
    df = df[df['clean_text'].str.strip() != '']

    return df


# ─────────────────────────────────────────────
# 5. ENCODE LABELS
# ─────────────────────────────────────────────

def encode_labels(train, dev, test, label_col):
    """
    SemEval labels might be 0/1 already, or strings like 'Good'/'Bad'.
    This handles both cases.
    """
    le = LabelEncoder()
    le.fit(train[label_col].astype(str))

    train['label_enc'] = le.transform(train[label_col].astype(str))
    dev['label_enc']   = le.transform(dev[label_col].astype(str))
    test['label_enc']  = le.transform(test[label_col].astype(str))

    print(f"\n=== Label Classes ===")
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    print(f"\nTrain label distribution:\n{train['label_enc'].value_counts()}")

    return train, dev, test, le


# ─────────────────────────────────────────────
# 6. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────

def vectorize(train, dev, test):
    vectorizer = TfidfVectorizer(
        max_features=10000,   # top 10k terms
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # apply log normalization
        min_df=2              # ignore terms appearing in <2 docs
    )

    X_train = vectorizer.fit_transform(train['clean_text'])
    X_dev   = vectorizer.transform(dev['clean_text'])
    X_test  = vectorizer.transform(test['clean_text'])

    y_train = train['label_enc'].values
    y_dev   = dev['label_enc'].values
    y_test  = test['label_enc'].values

    print(f"\n=== Vectorized Shapes ===")
    print(f"X_train : {X_train.shape}")
    print(f"X_dev   : {X_dev.shape}")
    print(f"X_test  : {X_test.shape}")

    return X_train, X_dev, X_test, y_train, y_dev, y_test, vectorizer


# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_artifacts(vectorizer, label_encoder, output_dir='artifacts'):
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"\nSaved vectorizer and label encoder to '{output_dir}/'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(train_path=None, dev_path=None, test_path=None):
    if train_path is None:
        train_path = r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-train.csv"
    if dev_path is None:
        dev_path = r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-dev.csv"
    if test_path is None:
        test_path = r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-test.csv"

    train, dev, test = load_data(train_path, dev_path, test_path)
    text_col, label_col = get_column_names(train)

    print("\n=== Cleaning text ===")
    train = preprocess_df(train, text_col, label_col)
    dev   = preprocess_df(dev,   text_col, label_col)
    test  = preprocess_df(test,  text_col, label_col)

    train, dev, test, label_encoder = encode_labels(train, dev, test, label_col)
    X_train, X_dev, X_test, y_train, y_dev, y_test, vectorizer = vectorize(train, dev, test)
    save_artifacts(vectorizer, label_encoder)

    return X_train, X_dev, X_test, y_train, y_dev, y_test, vectorizer, label_encoder

if __name__ == '__main__':
    main()