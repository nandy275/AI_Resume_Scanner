import joblib

SKILL_KEYWORDS = [
    "Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "Pandas", "NumPy", "Scikit-Learn", "SQL", "Power BI", "Tableau", "NLP",
    "HTML", "CSS", "JavaScript", "React", "Node.js", "MongoDB", "TypeScript",
    "Network Security", "Ethical Hacking", "Penetration Testing", "Kali Linux",
    "Wireshark", "SIEM", "Cryptography", "OWASP", "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", "Linux",
    "Accounting", "Tally", "GST", "Taxation", "Auditing", "Balance Sheet",
    "SAP FICO", "Financial Analysis", "Recruitment", "Talent Acquisition",
    "Payroll", "Onboarding", "HRIS", "Employee Engagement", "Training"
]


def predict_domain(resume_text):
  """Predicts the career domain and extracts matched skills."""
  try:
    model = joblib.load("resume_classifier_model.pkl")
    vectorizer = joblib.load("resume_tfidf_vectorizer.pkl")
  except FileNotFoundError:
    return "Error: Model files not found. Please run model_training.py first."

  X_input = vectorizer.transform([resume_text])
  predicted_domain = model.predict(X_input)[0]
  probabilities = model.predict_proba(X_input)[0]
  confidence = max(probabilities) * 100

  # Extract matching skills
  text_lower = resume_text.lower()
  detected_skills = [
      skill for skill in SKILL_KEYWORDS if skill.lower() in text_lower
  ]

  # Top domain probabilities breakdown
  classes = model.classes_
  top_matches = sorted(
      zip(classes, probabilities * 100), key=lambda x: x[1], reverse=True
  )[:3]

  return {
      "Domain": predicted_domain,
      "Confidence": f"{confidence:.2f}%",
      "Detected_Skills": detected_skills,
      "Top_Matches": top_matches,
  }


if __name__ == "__main__":
  sample = "Skilled in Python, Machine Learning, SQL, Pandas, and Deep Learning."
  print(predict_domain(sample))