# Pathology Analyzer

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-00A67E.svg)](https://openai.com/api/)
[![concurrent.futures](https://img.shields.io/badge/concurrent.futures-ThreadPoolExecutor-4B8BBE.svg)](https://docs.python.org/3/library/concurrent.futures.html)
[![tqdm](https://img.shields.io/badge/tqdm-v4.65.0-FFC107.svg)](https://github.com/tqdm/tqdm)
[![JSON](https://img.shields.io/badge/JSON-Output-000000.svg)](https://www.json.org/)
[![Typing](https://img.shields.io/badge/Typing-Annotations-3775A9.svg)](https://docs.python.org/3/library/typing.html)
[![Pathlib](https://img.shields.io/badge/Pathlib-Path-3775A9.svg)](https://docs.python.org/3/library/pathlib.html)
[![Logging](https://img.shields.io/badge/Logging-Standard-3775A9.svg)](https://docs.python.org/3/library/logging.html)

## Overview

The Pathology Analyzer is a robust system designed to extract, analyze, and structure information from pathology reports using advanced NLP techniques and medical knowledge bases. The system employs a sophisticated two-stage approach of extraction and verification, optimized for clinical accuracy and efficiency.

## Features

- **Text Extraction**: Concurrent processing of PDF pathology reports
- **Semantic Search**: Identification of relevant NCCN guideline chunks for context
- **LLM Integration**: Leverages OpenAI models for intelligent analysis with reasoning
- **Structured Data Extraction**: Extracts key cancer parameters (organ type, subtype, staging)
- **Verification System**: Two-stage verification process to ensure clinical accuracy
- **Batch Processing**: Support for processing multiple case folders efficiently
- **Statistical Analysis**: Comprehensive result collection and statistical summarization

## Architecture

The analysis module consists of these core components:

- **Process Pipeline**: Main workflow for analyzing pathology reports
- **Verification Agent**: Ensures extraction accuracy and consistency
- **Utilities**: Statistical analysis and parallel processing tools
- **Embedding System**: Semantic retrieval of relevant guidelines
- **Error Handling**: Comprehensive error management and logging

## Usage

```python
from config import AnalyzerConfig
from analysis.process import process_all_folders

# Configure the analyzer
config = AnalyzerConfig(
    root_dir="path/to/cases",
    openai_model="gpt-4o",
    openai_embed_model="text-embedding-3-large",
    enable_reasoning=True,
    verification_enabled=True
)

# Process all case folders
process_all_folders(config)

# Collect and analyze results
from analysis.utils import collect_results_parallel
summary = collect_results_parallel(config.root_dir)
```

## Key Parameters

- **text_similarity_threshold**: Threshold for text extraction similarity
- **k_guideline_chunks**: Number of guideline chunks to retrieve
- **sim_threshold**: Similarity threshold for guideline retrieval
- **enable_reasoning**: Toggle for advanced reasoning capabilities
- **verification_enabled**: Enable the two-stage verification process

## Output Format

The analysis produces a structured JSON output for each case with the following key fields:

- **cancer_organ_type**: Primary organ affected
- **cancer_subtype**: Specific cancer subtype
- **figo_stage**: FIGO staging classification
- **pathologic_stage**: Pathologic staging details
- **recommended_treatment**: Treatment recommendations based on guidelines
- **description**: Summary description of findings
- **patient_notes**: Notes for patient communication
- **verification**: Metadata about the verification process

## Performance Optimization

The system implements several optimizations:

- **Concurrent Text Extraction**: Multi-threaded PDF processing
- **Semantic Caching**: Embedding storage to avoid redundant computations
- **Parallel Result Collection**: Efficient multi-threaded result aggregation
- **Incremental Processing**: Skip-already-processed capability for large datasets

## Dependencies

The system relies on several Python packages:

- Core Python libraries (concurrent.futures, json, pathlib, logging)
- tqdm for progress monitoring
- OpenAI API for LLM capabilities and embeddings
- Custom modules for text extraction, embedding, and LLM interfacing

## License

MIT License

Copyright (c) 2025

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