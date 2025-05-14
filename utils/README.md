# Pathology Analyzer Utilities

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI_API-integrated-brightgreen)](https://openai.com/)
[![TikToken](https://img.shields.io/badge/TikToken-tokenizer-orange)](https://github.com/openai/tiktoken)
[![JSON](https://img.shields.io/badge/JSON-handling-lightgrey)](https://docs.python.org/3/library/json.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of robust utility modules that provide essential functionality for the Pathology Analyzer application. These utilities handle common tasks such as text processing, JSON manipulation, and logging.

## Modules

### `json.py`

Advanced JSON parsing and manipulation tools designed to handle potentially malformed JSON from LLM outputs.

**Key Features:**
- Robust JSON parsing with multiple fallback mechanisms
- Automatic repair of malformed JSON using OpenAI
- Regex-based JSON extraction from text blocks
- Simple interfaces for saving and loading JSON files

### `logger.py`

Comprehensive logging setup for the Pathology Analyzer application.

**Key Features:**
- Configurable logging to both console and files
- Timestamped log files for easy tracking
- Hierarchical logger organization
- Appropriate log level management for third-party libraries

### `text.py`

Text processing utilities focused on tokenization and text comparison.

**Key Features:**
- Token counting compatible with OpenAI models
- Text splitting based on token boundaries
- Text normalization for comparison
- Text similarity calculation using normalized Levenshtein distance
- Whitespace and formatting cleanup

## Installation

These utilities are designed to be used as part of the Pathology Analyzer application. No separate installation is required.

## Dependencies

- `openai`: For API access and JSON repair functionality
- `tiktoken`: For accurate token counting compatible with OpenAI models
- `logging`: Standard Python logging library
- `re`: Standard Python regex library
- `json`: Standard Python JSON library
- `pathlib`: Standard Python path manipulation library
- `difflib`: Standard Python text comparison library

## Usage

### JSON Handling

```python
from utils.json import safe_json, save_json, load_json

# Parse potentially malformed JSON
result = safe_json(llm_output_text)

# Save results to file
save_json(result, "analysis_results.json")

# Load data from file
data = load_json("config.json")
```

### Logging

```python
from utils.logger import setup_logging, get_logger
from pathlib import Path

# Set up logging for the application
setup_logging(Path("custom_logs"))

# Get a logger for a specific module
logger = get_logger("preprocessing")
logger.info("Starting preprocessing phase")
```

### Text Processing

```python
from utils.text import num_tokens, split_by_tokens, clean_text, text_similarity

# Check token count
tokens = num_tokens("Some text to analyze")

# Split long text into manageable chunks
chunks = split_by_tokens(long_document, max_tokens=4000)

# Clean and normalize text
clean = clean_text(raw_text)

# Compare text similarity
similarity = text_similarity(text1, text2)
```

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