import re
import pdfplumber
import docx
import spacy
import phonenumbers

nlp = spacy.load("en_core_web_sm")

# ------------ LOAD TEXT FROM FILE -------------- #
def load_text(path):
    if path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    elif path.lower().endswith(".docx") or path.lower().endswith(".doc"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif path.lower().endswith(".txt"):
        return open(path, "r", encoding="utf-8").read()

    return ""


# ------------ EXTRACT NAME (IMPROVED) ----------- #
def extract_name(text):
    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        # Skip empty or headers
        if not line or len(line.split()) > 5:
            continue

        # Detect line containing alphabetic chars only
        if re.match(r"^[A-Za-z][A-Za-z\s\.]+$", line):
            return line.title()

    # fallback NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    return None


# ------------ EXTRACT EMAIL --------------------- #
def extract_emails(text):
    return list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}", text)))


# ------------ EXTRACT PHONE --------------------- #
def extract_phones(text):
    phones = []
    for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
        phones.append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164))
    return list(set(phones))


# ------------ EXTRACT EDUCATION ----------------- #
def extract_education(text):
    edu_keywords = ["B.Tech", "B.E", "Bachelor", "Master", "M.Tech", "University", "College"]
    lines = text.split("\n")
    results = []
    for line in lines:
        if any(k.lower() in line.lower() for k in edu_keywords):
            results.append(line.strip())
    return results


# ------------ EXTRACT SKILLS -------------------- #
def extract_skills(text, skills_db):
    found = []
    for skill in skills_db:
        if skill.lower() in text.lower():
            found.append(skill)
    return sorted(list(set(found)))


# ------------ MAIN PARSE FUNCTION --------------- #
def parse_resume(path, skills_db):
    text = load_text(path)

    return {
        "name": extract_name(text),
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "education": extract_education(text),
        "skills": extract_skills(text, skills_db),
        "raw_text": text[:5000]   # limited to avoid overload
    }
