import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def preprocess_data(df):
  """Cleans data and extracts TF-IDF text features."""
  df = df.dropna(subset=["Resume_Text", "Domain"]).copy()

  X = df["Resume_Text"].astype(str)
  y = df["Domain"].astype(str)

  # Train-Test Split (Stratified)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # TF-IDF Vectorizer
  vectorizer = TfidfVectorizer(
      max_features=2000, ngram_range=(1, 2), stop_words="english"
  )
  X_train_tfidf = vectorizer.fit_transform(X_train)
  X_test_tfidf = vectorizer.transform(X_test)

  print(f"Training features shape: {X_train_tfidf.shape}")
  print(f"Testing features shape: {X_test_tfidf.shape}")

  return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


if __name__ == "__main__":
  from read_dataset import load_dataset

  df = load_dataset()
  if df is not None:
    preprocess_data(df)