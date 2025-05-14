from config import AnalyzerConfig, ORGAN_TYPES_STR, SUBTYPES_STR
from text_extraction.extract import extract_text_concurrent
from embedding.embed import load_or_create_embeddings
from analysis.process import analyse_report, process_all_folders
from llm.verification import VerificationAgent
from exceptions import (
    PathologyAnalyzerError, TextExtractionError, OpenAIAPIError,
    EmbeddingError, VerificationError, TextComparisonError
)

__version__ = "1.0.0"
