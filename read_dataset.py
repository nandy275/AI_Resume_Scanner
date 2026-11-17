from pathlib import Path
import pandas as pd

DATASET_PATH = Path("dataset/Resume_Scanner_Dataset_5000.xlsx")


def load_dataset(file_path=DATASET_PATH):
  """Loads the 5000-resume dataset from the Excel file."""
  try:
    df = pd.read_excel(file_path, sheet_name="Resume_Dataset")
    print(f"Dataset successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df
  except Exception as e:
    print(f"Error loading dataset: {e}")
    return None


if __name__ == "__main__":
  df = load_dataset()
  if df is not None:
    print(df.head(2))