"""
text_extraction/extract.py - Text extraction functions for the Pathology Analyzer application
"""

import io
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.text import text_similarity, clean_text
from exceptions import TextExtractionError, TextComparisonError

logger = logging.getLogger("pathology_analyzer.text_extraction")


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz).

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        str: Extracted text

    Raises:
        Exception: If extraction fails
    """
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
            return clean_text(text)
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        raise


def extract_text_pypdf2(pdf_path: Path) -> str:
    """Extract text from PDF using PyPDF2.

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        str: Extracted text

    Raises:
        Exception: If extraction fails
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return clean_text(text)
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
        raise


def extract_text_ocr(pdf_path: Path, dpi: int = 400) -> str:
    """Extract text from PDF using OCR via pytesseract.

    Args:
        pdf_path (Path): Path to the PDF file
        dpi (int): DPI for rendering PDF pages as images

    Returns:
        str: Extracted text

    Raises:
        Exception: If extraction fails
    """
    logger.info(f"Performing OCR on {pdf_path.name} at {dpi} DPI...")
    text_pages = []

    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                logger.info(f"Processing page {i + 1}/{len(doc)}")
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes()))
                text_pages.append(pytesseract.image_to_string(img))

        text = "\n".join(text_pages)
        return clean_text(text)
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((IOError, ValueError))
)
def extract_text_concurrent(pdf_path: Path, similarity_threshold: float = 0.8) -> str:
    """Extract text from PDF with concurrent methods and comparison.

    This function uses both PyMuPDF and PyPDF2 concurrently to extract text from a PDF.
    If the results are sufficiently similar (above the similarity threshold), it returns
    the PyMuPDF result (generally higher quality). Otherwise, it falls back to OCR.

    Args:
        pdf_path (Path): Path to the PDF file
        similarity_threshold (float): Threshold for comparing results from different extractors

    Returns:
        str: Extracted text

    Raises:
        TextExtractionError: If all extraction methods fail
    """
    logger.info(f"Extracting text from {pdf_path.name} using concurrent methods")

    # Define a function to run extraction methods with proper exception handling
    def run_extractor(extractor_func):
        try:
            return extractor_func(pdf_path)
        except Exception as e:
            logger.warning(f"{extractor_func.__name__} failed: {e}")
            return None

    # Run both extractors concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_pymupdf = executor.submit(run_extractor, extract_text_pymupdf)
        future_pypdf2 = executor.submit(run_extractor, extract_text_pypdf2)

        pymupdf_text = future_pymupdf.result()
        pypdf2_text = future_pypdf2.result()

    # If both methods succeeded, compare results
    if pymupdf_text and pypdf2_text:
        if len(pymupdf_text) > 100 and len(pypdf2_text) > 100:
            similarity = text_similarity(pymupdf_text, pypdf2_text)
            logger.info(f"Text extraction similarity: {similarity:.2f}")

            if similarity >= similarity_threshold:
                logger.info("Extraction methods show high agreement, using PyMuPDF result")
                return pymupdf_text
            else:
                logger.warning(f"Low similarity ({similarity:.2f}) between extraction methods, falling back to OCR")
        else:
            logger.warning("One or both extraction methods returned minimal text, falling back to OCR")
    elif pymupdf_text and len(pymupdf_text) > 100:
        logger.info("Only PyMuPDF extraction succeeded with meaningful text")
        return pymupdf_text
    elif pypdf2_text and len(pypdf2_text) > 100:
        logger.info("Only PyPDF2 extraction succeeded with meaningful text")
        return pypdf2_text

    # Fall back to OCR
    try:
        ocr_text = extract_text_ocr(pdf_path)
        if len(ocr_text) > 100:
            logger.info("Successfully extracted text using OCR")
            return ocr_text
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")

    # If all methods fail
    raise TextExtractionError(f"Could not extract meaningful text from {pdf_path.name} using any method")


def compare_extraction_methods(pdf_path: Path) -> Dict[str, Tuple[str, int]]:
    """Compare different text extraction methods on a PDF.

    This function runs all text extraction methods on the same PDF and
    returns the results along with statistics for comparison.

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        Dict[str, Tuple[str, int]]: Dictionary mapping method names to tuples of
            (extracted_text, character_count)
    """
    results = {}

    try:
        pymupdf_text = extract_text_pymupdf(pdf_path)
        results["pymupdf"] = (pymupdf_text, len(pymupdf_text))
    except Exception:
        results["pymupdf"] = ("FAILED", 0)

    try:
        pypdf2_text = extract_text_pypdf2(pdf_path)
        results["pypdf2"] = (pypdf2_text, len(pypdf2_text))
    except Exception:
        results["pypdf2"] = ("FAILED", 0)

    try:
        ocr_text = extract_text_ocr(pdf_path)
        results["ocr"] = (ocr_text, len(ocr_text))
    except Exception:
        results["ocr"] = ("FAILED", 0)

    # Calculate similarity if all methods succeeded
    if "FAILED" not in [v[0] for v in results.values()]:
        results["pymupdf_vs_pypdf2"] = (
            f"Similarity: {text_similarity(results['pymupdf'][0], results['pypdf2'][0]):.2f}",
            0
        )
        results["pymupdf_vs_ocr"] = (
            f"Similarity: {text_similarity(results['pymupdf'][0], results['ocr'][0]):.2f}",
            0
        )
        results["pypdf2_vs_ocr"] = (
            f"Similarity: {text_similarity(results['pypdf2'][0], results['ocr'][0]):.2f}",
            0
        )

    return results