# Embedding

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green)](https://openai.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange)](https://numpy.org/)
[![Tenacity](https://img.shields.io/badge/Tenacity-8.0%2B-brightgreen)](https://github.com/jd/tenacity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust text embedding module for pathology report analysis that leverages OpenAI's embedding models for semantic search and information retrieval.

## Overview

This module contains utilities for creating and managing embeddings from text data, with a focus on medical pathology reports and NCCN guidelines. It provides functionality for:

- Converting text to vector embeddings using OpenAI's state-of-the-art models
- Efficient caching and persistence of computed embeddings
- Semantic similarity search across medical guidelines
- Robust error handling and automatic retries for API requests

## Files

- `embed.py`: Core embedding functions and caching utilities
- `retrieval.py`: Semantic search and similarity functions

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy openai tenacity tqdm
```

## Usage

### Basic Embedding

```python
from embedding.embed import embed

# Create an embedding for a text string
text = "Invasive ductal carcinoma, grade 2, ER+/PR+, HER2-"
embedding_vector = embed(text)

# The resulting vector can be used for similarity comparisons
```

### Guideline Retrieval

```python
from embedding.retrieval import top_k_guideline
from embedding.embed import load_or_create_embeddings
from config import AnalyzerConfig

# Load configuration
config = AnalyzerConfig()

# Load or create guideline embeddings
guide_chunks, guide_embeds = load_or_create_embeddings(config)

# Find relevant guideline sections for a pathology report
report_text = "Patient presents with invasive ductal carcinoma..."
relevant_sections = top_k_guideline(
    query=report_text,
    guide_chunks=guide_chunks,
    guide_embeds=guide_embeds,
    k=5,  # Return top 5 most relevant sections
    threshold=0.3  # Minimum similarity score
)

# Process or display the relevant guideline sections
for i, section in enumerate(relevant_sections, 1):
    print(f"Relevant section {i}:\n{section}\n")
```

### Clearing the Cache

```python
from embedding.embed import clear_embedding_cache

# Clear the embedding cache when needed
clear_embedding_cache()
```

## Key Features

### Optimized Performance

- **LRU Caching**: Automatically caches embeddings to avoid redundant API calls
- **Concurrent Text Extraction**: Efficiently processes large PDF documents
- **Persistent Storage**: Saves computed embeddings to disk for future use

### Robustness

- **Automatic Retries**: Uses exponential backoff for transient API failures
- **Comprehensive Logging**: Detailed logging of embedding operations
- **Error Handling**: Custom exceptions for clear error identification

### Semantic Search

- **Cosine Similarity**: Measures semantic relatedness between texts
- **Configurable Thresholds**: Adjustable similarity thresholds for precision control
- **Top-K Retrieval**: Returns the most relevant document chunks

## Configuration

The module uses the `AnalyzerConfig` class for configuration parameters:

- `openai_embed_model`: The OpenAI embedding model to use (default: "text-embedding-3-large")
- `embeddings_cache_path`: Path for caching embeddings
- `chunk_token_size`: Token size for chunking guidelines
- `text_similarity_threshold`: Threshold for deduplicating similar text segments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
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
```