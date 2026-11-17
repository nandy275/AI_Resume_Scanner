import joblib
from preprocessing import preprocess_data
from read_dataset import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model():
  """Evaluates the trained model using test data."""
  df = load_dataset()
  if df is None:
    return

  _, X_test, _, y_test, _ = preprocess_data(df)

  try:
    model = joblib.load("resume_classifier_model.pkl")
  except FileNotFoundError:
    print("Model file not found. Please run model_training.py first.")
    return

  y_pred = model.predict(X_test)

  print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
  print("Classification Report:")
  print(classification_report(y_test, y_pred))

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
  evaluate_model()