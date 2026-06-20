# io_utils.py
# Utilities to read text from PDF, DOCX, and TXT files.

import pdfplumber
import docx
from pathlib import Path

def read_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return '\n'.join(paragraphs)

def read_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)

def load_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in ['.txt', '.md']:
        return read_txt(path)
    if p.suffix.lower() in ['.docx']:
        return read_docx(path)
    if p.suffix.lower() in ['.pdf']:
        return read_pdf(path)
    # fallback: try reading as text
    return read_txt(path)
