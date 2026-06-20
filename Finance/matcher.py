# matcher.py
# Create embeddings for resume text and job description and compute cosine similarity.

from sentence_transformers import SentenceTransformer, util
import json
from parser import parse_resume

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def resume_embedding(parsed_resume):
    # Use summary + skills + experience header + education as combined text
    parts = []
    if parsed_resume.get("summary"):
        parts.append(parsed_resume["summary"])
    parts.extend(parsed_resume.get("skills", []))
    # add experience headers
    for job in parsed_resume.get("experience", [])[:5]:
        parts.append(job.get("header",""))
    if parsed_resume.get("education"):
        parts.append(' '.join(parsed_resume['education']))
    text = ' '.join(parts)
    return model.encode(text, convert_to_tensor=True)

def job_embedding(job_text):
    return model.encode(job_text, convert_to_tensor=True)

def score_resume_against_job(resume_path, job_text, skills_db):
    parsed = parse_resume(resume_path, skills_db)
    emb_r = resume_embedding(parsed)
    emb_j = job_embedding(job_text)
    cos = util.cos_sim(emb_r, emb_j).item()
    return cos, parsed

if __name__ == "__main__":
    import sys
    skills_db = json.load(open("skills_db.json"))
    resume_path = sys.argv[1] if len(sys.argv)>1 else "examples/resume_1.txt"
    job_text = """Seeking an AI developer with experience in computer vision, PyTorch, OpenCV, and model deployment.
    Responsibilities include building inference pipelines and integrating models into web apps."""
    score, parsed = score_resume_against_job(resume_path, job_text, skills_db)
    print("Score:", score)
    import pprint
    pprint.pprint(parsed)
