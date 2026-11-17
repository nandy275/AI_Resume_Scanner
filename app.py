import streamlit as st
from prediction import predict_domain

st.set_page_config(
    page_title="AI-Based Resume Scanner", page_icon="📄", layout="centered"
)

st.title("📄 AI-Based Resume Scanner")
st.write(
    "Paste your resume text or key skills below to classify the candidate's"
    " **Career Domain** and extract core skills!"
)

resume_input = st.text_area(
    "Paste Resume Summary / Skills:",
    height=180,
    placeholder=(
        "Example: Experienced in Python, Machine Learning, SQL, Pandas, and"
        " TensorFlow with 3 years of experience..."
    ),
)

if st.button("Scan & Classify Resume"):
  if not resume_input or resume_input.strip() == "":
    st.warning("⚠️ Please enter resume text or skills before scanning.")
  else:
    with st.spinner("Analyzing resume..."):
      result = predict_domain(resume_input)

      if isinstance(result, str):
        st.error(result)
      else:
        st.subheader("Scan Results")
        st.success(
            f"**Predicted Domain:** {result['Domain']} (Confidence:"
            f" {result['Confidence']})"
        )

        if result["Detected_Skills"]:
          st.write(
              "**Extracted Skills:** " + ", ".join(result["Detected_Skills"])
          )
        else:
          st.write("**Extracted Skills:** No standard keywords matched.")

        st.write("**Top Domain Confidence Breakdown:**")
        for dom, score in result["Top_Matches"]:
          st.progress(int(score), text=f"{dom}: {score:.1f}%")