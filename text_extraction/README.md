# Text Extraction Module

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.19+-green.svg)](https://pymupdf.readthedocs.io/)
[![PyPDF2](https://img.shields.io/badge/PyPDF2-2.0+-yellow.svg)](https://pypdf2.readthedocs.io/)
[![pytesseract](https://img.shields.io/badge/pytesseract-0.3+-red.svg)](https://pypi.org/project/pytesseract/)
[![PIL](https://img.shields.io/badge/Pillow-8.0+-purple.svg)](https://pillow.readthedocs.io/)

A robust text extraction library designed for the Pathology Analyzer application, providing multiple methods to extract text from PDF documents with fallback mechanisms and quality comparison.

## Overview

This module employs three different text extraction methods (PyMuPDF, PyPDF2, and OCR via pytesseract) to reliably extract text from PDF files, even when dealing with problematic documents. It includes sophisticated comparison algorithms to select the best extraction result or to fall back to OCR when needed.

## Features

- **Multiple Extraction Methods**: Uses PyMuPDF (fitz), PyPDF2, and OCR via pytesseract
- **Concurrent Processing**: Performs parallel extraction to improve performance
- **Smart Comparison**: Compares extraction results to identify the most accurate text
- **Fallback Mechanism**: Automatically falls back to OCR when other methods fail or produce inconsistent results
- **Quality Analysis**: Provides detailed analysis of differences between extraction methods
- **Resilient Operation**: Includes retry mechanisms for handling intermittent failures

## Installation

```bash
# Install the module and its dependencies
pip install -r requirements.txt
```

## Usage

### Basic Text Extraction

```python
from pathlib import Path
from text_extraction.extract import extract_text_concurrent

# Extract text using the best available method
pdf_path = Path("path/to/document.pdf")
extracted_text = extract_text_concurrent(pdf_path)
print(f"Extracted {len(extracted_text)} characters of text")
```

### Comparing Extraction Methods

```python
from text_extraction.extract import compare_extraction_methods

# Compare results from different extraction methods
pdf_path = Path("path/to/document.pdf")
comparison_results = compare_extraction_methods(pdf_path)

# Print results
for method, (text, char_count) in comparison_results.items():
    if not method.startswith("pymupdf_vs"):
        print(f"{method}: {char_count} characters")
    else:
        print(f"{method}: {text}")
```

### Analyzing Text Differences

```python
from text_extraction.compare import analyze_text_differences

# Analyze differences between two extracted texts
text1 = "Sample text from first method"
text2 = "Sample text from second method"
analysis = analyze_text_differences(text1, text2)

print(f"Similarity score: {analysis['similarity_score']:.2f}")
print(f"Character count difference: {analysis['character_count']['difference']}")
```

## Module Contents

### extract.py

Contains functions for extracting text from PDF files using different methods:

- `extract_text_pymupdf`: Extracts text using PyMuPDF
- `extract_text_pypdf2`: Extracts text using PyPDF2
- `extract_text_ocr`: Extracts text using OCR via pytesseract
- `extract_text_concurrent`: Extracts text using multiple methods concurrently and selects the best result
- `compare_extraction_methods`: Compares the results of different extraction methods

### compare.py

Contains functions for comparing and analyzing extracted texts:

- `compare_extracted_texts`: Compares two extracted texts and determines if they are sufficiently similar
- `analyze_text_differences`: Analyzes differences between two extracted texts
- `select_best_text`: Selects the best text from multiple extraction methods

## Dependencies

- **PyMuPDF (fitz)**: High-performance PDF processing
- **PyPDF2**: Pure-Python PDF processing
- **pytesseract**: Python wrapper for Tesseract OCR
- **Pillow (PIL)**: Image processing library
- **tenacity**: Retry mechanism for handling failures
- **concurrent.futures**: Parallel processing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Pathology Analyzer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```