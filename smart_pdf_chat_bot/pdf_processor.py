"""
PDF Processing Module
Uses Mistral OCR API for robust text extraction from all PDF types
(including image-based/scanned PDFs).
"""

import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from mistralai.client.sdk import Mistral
from mistralai.client.models.documenturlchunk import DocumentURLChunk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def _extract_text_with_mistral_ocr(pdf_path):
    """
    Extract text from a PDF using Mistral OCR API.

    Pipeline:
      1. Upload PDF to Mistral
      2. Get signed URL
      3. Run OCR → returns JSON with per-page markdown

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        Tuple of (list of LangChain Document objects, raw OCR JSON response dict)
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is not set. Add it to your .env file."
        )

    client = Mistral(api_key=MISTRAL_API_KEY)
    pdf_path = Path(pdf_path)

    # Step 1: Upload PDF to Mistral
    print(f"[Mistral OCR] Uploading '{pdf_path.name}' to Mistral...")
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_path.stem,
            "content": pdf_path.read_bytes(),
        },
        purpose="ocr",
    )
    print(f"[Mistral OCR] Upload complete. File ID: {uploaded_file.id}")

    # Step 2: Get signed URL
    print("[Mistral OCR] Getting signed URL...")
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    # Step 3: Run OCR
    print("[Mistral OCR] Running OCR processing...")
    ocr_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=False,  # We only need text for RAG
    )

    response_dict = json.loads(ocr_response.model_dump_json())
    pages = response_dict.get("pages", [])
    print(f"[Mistral OCR] OCR complete. Extracted {len(pages)} page(s).")

    # Step 4: Convert to LangChain Document objects
    documents = []
    for page_idx, page in enumerate(pages):
        markdown_text = page.get("markdown", "").strip()

        if markdown_text:
            documents.append(
                Document(
                    page_content=markdown_text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_idx,
                        "page_label": str(page_idx + 1),
                        "total_pages": len(pages),
                    },
                )
            )

        char_count = len(markdown_text)
        preview = markdown_text[:100].replace("\n", " ") if markdown_text else "(empty)"
        print(f"[Mistral OCR] Page {page_idx + 1}: {char_count} chars — {preview}...")

    return documents, response_dict


def load_and_split(uploaded_file, chunk_size=1000, chunk_overlap=200):
    """
    Read an uploaded PDF file, extract text via Mistral OCR, and split into chunks.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of LangChain Document objects (chunks).
    """
    # Create a temp directory and write the file there
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "upload.pdf")

    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        documents, ocr_json = _extract_text_with_mistral_ocr(tmp_path)

        # Debug summary
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"[PDF Processor] Total: {len(documents)} pages, {total_chars} characters extracted")

        if total_chars == 0:
            print("[PDF Processor] WARNING: No text extracted from PDF!")
            return []

        # Split into chunks for embedding
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"[PDF Processor] Split into {len(chunks)} chunks")

        return chunks
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass
