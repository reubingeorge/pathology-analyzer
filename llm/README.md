# Pathology Analyzer LLM Module

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-7B83EB)](https://openai.com/)
[![Tenacity](https://img.shields.io/badge/Tenacity-8.2.2-orange)](https://github.com/jd/tenacity)
[![JSON Schema](https://img.shields.io/badge/JSON-Schema-lightgrey)](https://json-schema.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The LLM module provides a robust interface for analyzing oncology pathology reports using OpenAI's advanced language models. This module extracts structured medical information according to NCCN guidelines, performs verification and validation, and implements medical inference to ensure high-quality results.

## Features

- **Optimized OpenAI API Client**: Efficient, retry-capable communication with OpenAI's API
- **Reasoning-Enhanced Prompting**: Step-by-step reasoning capabilities for improved medical analysis
- **Comprehensive Verification**: Multi-stage verification system for ensuring medical accuracy
- **Medical Inference Validation**: Cross-checks data consistency against medical knowledge
- **Error Handling**: Robust error management with intelligent fallbacks
- **JSON Output**: Standardized structured data output with medical terminology

## Files

### `client.py`

Provides a specialized OpenAI API client with the following capabilities:
- Retry logic with exponential backoff for API resilience
- Reasoning-enhanced prompt optimization
- Structured JSON response handling
- Custom temperature management optimized for different models

```python
# Example usage
from llm.client import analyze_with_reasoning

result = analyze_with_reasoning(
    msgs=[
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt}
    ],
    model="o4-mini",
    enable_reasoning=True
)
```

### `prompts.py`

Contains a comprehensive set of specialized prompts for oncology report analysis:
- System messages with strict formatting requirements
- Step-by-step reasoning instructions for complex medical analysis
- Verification prompts with focus on staging accuracy
- Medical inference validation templates
- Structured JSON output specifications

### `verification.py`

Implements a sophisticated verification system with:
- Combined verification approach to reduce API calls
- Focused validation on cancer subtype and staging information
- Medical inference validation for consistency checking
- Detailed correction application with explanations
- Comprehensive verification metadata

```python
# Example usage
from config import AnalyzerConfig
from llm.verification import VerificationAgent

config = AnalyzerConfig(openai_model="o4-mini", enable_reasoning=True)
verifier = VerificationAgent(config)
verification_result = verifier.verify(report_text, extraction, nccn_text)
corrected_data = verifier.apply_corrections(extraction, verification_result)
```

## Key Components

### Enhanced Reasoning System

The module implements a sophisticated reasoning system that encourages the language model to think step-by-step when analyzing medical reports, significantly improving accuracy in complex medical tasks.

### Verification Pipeline

The verification pipeline includes:
1. Primary extraction verification with focus on staging
2. Medical inference validation for consistency
3. Application of corrections with detailed explanations
4. Strict validation against allowed medical terminology

### Error Handling and Resilience

The module features robust error handling with:
- Automatic retries with exponential backoff
- Fallback strategies for API failures
- Validation against missing or invalid data
- Detailed logging for troubleshooting

## Installation

This module requires Python 3.8+ and the following dependencies:
- openai
- tenacity
- logging (standard library)

```bash
pip install openai tenacity
```

## Usage

```python
from llm.client import analyze_with_reasoning
from llm.verification import VerificationAgent
from config import AnalyzerConfig

# Initialize configuration
config = AnalyzerConfig(
    openai_model="o4-mini",
    enable_reasoning=True,
    detailed_verification=True
)

# Analyze pathology report
analysis_result = analyze_with_reasoning(
    msgs=[
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": formatted_prompt}
    ],
    model=config.openai_model,
    enable_reasoning=config.enable_reasoning
)

# Verify results
verifier = VerificationAgent(config)
verification = verifier.verify(report_text, analysis_result, nccn_guidelines)
final_result = verifier.apply_corrections(analysis_result, verification)
```

## License

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