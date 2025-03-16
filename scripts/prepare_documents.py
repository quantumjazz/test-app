import os
import re
import csv
import glob
import sys
import PyPDF2
import docx

# Optionally, if you want more robust tokenization you could uncomment the next lines:
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

def read_settings(settings_path: str) -> dict:
    """
    Reads simple key-value pairs from a settings.txt file.
    Expected format (one key=value per line).
    """
    settings = {}
    with open(settings_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            settings[key.strip()] = value.strip()
    return settings

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDF2.
    """
    text_content = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    """
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text:
            full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_txt(txt_path: str) -> str:
    """
    Extracts text from a TXT file.
    """
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_text_from_tex(tex_path: str) -> str:
    """
    Extracts text from a TEX file by reading the raw text.
    (Further cleaning could be added if needed.)
    """
    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 100, title: str = "") -> list:
    """
    Splits text into chunks of roughly `chunk_size` words with an `overlap`.
    Each chunk is prefixed with the document title for context.
    
    You can adjust chunk_size and overlap as needed.
    """
    # If you prefer robust tokenization, replace the next line with:
    # words = word_tokenize(text)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        # Add a prefix with the document title if provided
        if title:
            chunk = f"This text comes from the document {title}. " + chunk
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def main():
    """
    Main routine to process course documents:
      1. Read settings from settings.txt.
      2. Recursively gather documents from the specified documents folder.
      3. Split each document's text into overlapping chunks.
      4. Write all chunks to a single CSV file in the 'data/' folder.
    """
    # Determine base directory (assumes this script is in 'scripts/')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    settings_path = os.path.join(base_dir, 'settings.txt')
    settings = read_settings(settings_path)
    print("Loaded settings:", settings)
    
    # Use the 'filedirectory' from settings (default to 'documents' if not specified)
    documents_dir = os.path.join(base_dir, settings.get("filedirectory", "documents"))
    output_csv_path = os.path.join(base_dir, 'data', 'chopped_text.csv')
    
    # Optional: allow chunking parameters to be set in settings.txt
    try:
        chunk_size = int(settings.get('chunk_size', 200))
        overlap = int(settings.get('overlap', 100))
    except ValueError:
        chunk_size = 200
        overlap = 100

    # 2. Gather files (PDF, DOCX, TXT, TEX) from the documents directory
    file_patterns = [
        os.path.join(documents_dir, '**', '*.pdf'),
        os.path.join(documents_dir, '**', '*.docx'),
        os.path.join(documents_dir, '**', '*.txt'),
        os.path.join(documents_dir, '**', '*.tex'),
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        print(f"No documents found in {documents_dir}. Exiting.")
        sys.exit(0)

    # 3. Process and chunk each file
    all_chunks = []  # Will store tuples: (filename, chunk_index, chunk_text)
    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        print(f"Processing: {fpath}")
        text = ""
        if ext == '.pdf':
            text = extract_text_from_pdf(fpath)
        elif ext == '.docx':
            text = extract_text_from_docx(fpath)
        elif ext == '.txt':
            text = extract_text_from_txt(fpath)
        elif ext == '.tex':
            text = extract_text_from_tex(fpath)
        else:
            print(f"Skipping unsupported file: {fpath}")
            continue

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split text into chunks and add a title prefix
        filename_only = os.path.basename(fpath)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, title=filename_only)

        for i, chunk in enumerate(chunks):
            all_chunks.append((filename_only, i, chunk))

    # 4. Write all chunks to CSV in the 'data/' folder
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "chunk_index", "chunk_text"])
        for entry in all_chunks:
            writer.writerow(entry)

    print(f"Done! Wrote {len(all_chunks)} total chunks to {output_csv_path}")

if __name__ == "__main__":
    main()
