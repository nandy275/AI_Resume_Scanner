import matplotlib.pyplot as plt
import pandas as pd
from read_dataset import load_dataset


def perform_eda(df):
  """Performs Exploratory Data Analysis and prints summary statistics."""
  print("--- Dataset Info ---")
  print(df.info())

  print("\n--- Missing Values ---")
  print(df.isnull().sum())

  print("\n--- Domain Distribution ---")
  print(df["Domain"].value_counts())

  print("\n--- Experience Years Summary ---")
  print(df["Experience_Years"].describe())

  # Visualizing Domain Distribution
  df["Domain"].value_counts().plot(
      kind="bar",
      color="#1E3A5F",
      title="Resume Distribution Across Career Domains",
  )
  plt.xlabel("Domain")
  plt.ylabel("Count")
  plt.xticks(rotation=30, ha="right")
  plt.tight_layout()
  plt.savefig("domain_distribution.png")
  plt.close()
  print("\nSaved domain distribution chart to 'domain_distribution.png'.")


if __name__ == "__main__":
  df = load_dataset()
  if df is not None:
    perform_eda(df)