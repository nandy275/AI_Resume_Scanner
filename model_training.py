import joblib
from preprocessing import preprocess_data
from read_dataset import load_dataset
from sklearn.linear_model import LogisticRegression


def train_model():
  """Trains the resume domain classifier and saves artifacts."""
  df = load_dataset()
  if df is None:
    return

  X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

  model = LogisticRegression(max_iter=1000, random_state=42)
  model.fit(X_train, y_train)

  joblib.dump(model, "resume_classifier_model.pkl")
  joblib.dump(vectorizer, "resume_tfidf_vectorizer.pkl")
  print("Model and vectorizer successfully trained and saved!")

  return model, vectorizer, X_test, y_test


if __name__ == "__main__":
  train_model()