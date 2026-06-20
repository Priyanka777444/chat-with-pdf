# app.py
# Run with: streamlit run app.py

import streamlit as st
from parser import parse_resume
from matcher import score_resume_against_job
import json
from pathlib import Path

st.set_page_config(page_title="Resume Parser + Matcher", layout="wide")
st.title("Resume Parser + Job Matcher ")

# Sidebar: job description
job_desc = st.sidebar.text_area(
    "Paste a job description to score resumes against:",
    value="""Seeking an AI developer with experience in computer vision, PyTorch, OpenCV, and model deployment.
Responsibilities include building inference pipelines and integrating models into web apps.""",
    height=200,
)

uploaded_file = st.file_uploader("Upload resume (.txt, .pdf, .docx)", type=['pdf','txt','docx'])
if uploaded_file is not None:
    # save temporarily
    tpath = Path("tmp_upload") / uploaded_file.name
    tpath.parent.mkdir(exist_ok=True)
    with open(tpath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved {uploaded_file.name}")
    # load skills_db
    skills_db = json.load(open("skills_db.json"))
    with st.spinner("Parsing resume..."):
        parsed = parse_resume(str(tpath), skills_db)
    st.subheader("Parsed JSON")
    st.json(parsed)
    st.subheader("Top-level summary")
    st.write(f"Name: **{parsed.get('name')}**")
    st.write(f"Emails: {parsed.get('emails')}")
    st.write(f"Phones: {parsed.get('phones')}")
    st.write("Skills:", ', '.join(parsed.get('skills') or []))
    st.write("Education:", parsed.get('education'))
    st.markdown("---")
    st.subheader("Score vs. Job Description")
    if job_desc.strip():
        with st.spinner("Computing similarity..."):
            score, _ = score_resume_against_job(str(tpath), job_desc, skills_db)
        st.metric(label="Similarity Score (0-1)", value=f"{score:.3f}")
    else:
        st.info("Add a job description in the sidebar to compute similarity score.")

else:
    st.info("Upload a resume to begin. You can use the example resumes in the `examples/` folder.")
    if st.button("Load Example 1"):
        st.experimental_rerun()
