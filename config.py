"""
config.py - Configuration class and settings for the Pathology Analyzer application
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalyzerConfig:
    """Configuration for the Pathology Analyzer.

    This dataclass contains all configurable parameters for the Pathology Analyzer application,
    including API credentials, model selections, file paths, and algorithm parameters.
    It provides default values for all fields and a validation method to ensure the
    configuration is valid before running the analyzer.

    Attributes:
        openai_api_key (str): OpenAI API key for authentication, defaults to environment variable
        openai_model (str): Main OpenAI model for report analysis, defaults to "o4-mini"
        openai_embed_model (str): OpenAI model for creating embeddings, defaults to "text-embedding-3-large"
        openai_repair_model (str): OpenAI model for JSON repair, defaults to "gpt-3.5-turbo"
        root_dir (Path): Root directory containing case folders, defaults to "cases"
        nccn_pdf_path (Path): Path to NCCN guidelines PDF, defaults to "uterine_core.pdf"
        embeddings_cache_path (Path): Path for cached embeddings, defaults to "nccn_embeddings_cache.pkl"
        log_dir (Path): Directory for log files, defaults to "logs"
        chunk_token_size (int): Size of text chunks for embedding, defaults to 500
        k_guideline_chunks (int): Number of guideline chunks to retrieve, defaults to 8
        sim_threshold (float): Minimum similarity threshold for chunks, defaults to 0.25
        max_total_tokens (int): Maximum token limit for API calls, defaults to 110,000
        verification_enabled (bool): Whether to verify extracted data, defaults to True
        verification_threshold (float): Confidence threshold for verification, defaults to 0.8
        detailed_verification (bool): Whether to perform detailed field verification, defaults to True
        text_similarity_threshold (float): Threshold for text extraction similarity, defaults to 0.8
        enable_reasoning (bool): Whether to enable reasoning for OpenAI models, defaults to True
    """
    # API Key and Models
    openai_api_key: str = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
    openai_model: str = "o4-mini"  # Default model for main analysis
    openai_embed_model: str = "text-embedding-3-large"  # Default model for embeddings
    openai_repair_model: str = "gpt-3.5-turbo"  # Default model for JSON repair

    # Paths
    root_dir: Path = Path("cases")  # Path to case directories
    nccn_pdf_path: Path = Path("uterine_core.pdf")  # Path to guidelines PDF
    embeddings_cache_path: Path = Path("nccn_embeddings_cache.pkl")  # Path for cache
    log_dir: Path = Path("logs")  # Path for log files

    # Extraction parameters
    chunk_token_size: int = 500  # Size of chunks for tokenization
    k_guideline_chunks: int = 8  # Number of chunks to retrieve
    sim_threshold: float = 0.25  # Similarity threshold for relevance
    max_total_tokens: int = 110_000  # Max tokens for API calls

    # Verification parameters
    verification_enabled: bool = True  # Enable/disable verification
    verification_threshold: float = 0.8  # Confidence threshold
    detailed_verification: bool = True  # Enable detailed verification

    # Text extraction parameters
    text_similarity_threshold: float = 0.8  # Threshold for text extraction similarity

    # Reasoning parameters
    enable_reasoning: bool = True  # Enable reasoning for OpenAI models

    def validate(self):
        """Validate the configuration settings.

        Checks that all required configuration values are present and valid.
        Currently verifies that the OpenAI API key is set.

        Returns:
            self: Returns the validated configuration object for method chaining

        Raises:
            ValueError: If the OpenAI API key is missing or any other validation fails
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Ensure all directories exist
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.root_dir.mkdir(exist_ok=True, parents=True)

        # Verify NCCN PDF exists
        if not self.nccn_pdf_path.exists():
            logging.warning(f"NCCN PDF not found at {self.nccn_pdf_path}. Make sure to provide this file.")

        return self


# Constants for cancer types
ORGAN_TYPES_STR = (
    "Bladder cancer, Blood/Lymph Cancer, Brain cancer, Breast cancer, Cervical cancer, "
    "Colorectal cancer, Esophageal cancer, Kidney cancer, Liver Cancer, Lung cancer, "
    "Ovarian cancer, Pancreatic cancer, Prostate cancer, Skin cancer, Stomach cancer, "
    "Thyroid cancer, Uterine cancer"
)

# List of known cancer subtypes for each organ
SUBTYPES_STR = """ - Bladder cancer: No specific subtypes
 - Blood/Lymph Cancer: No specific subtypes
 - Brain cancer: No specific subtypes
 - Breast cancer: No specific subtypes
 - Cervical cancer: No specific subtypes
 - Colorectal cancer: No specific subtypes
 - Esophageal cancer: No specific subtypes
 - Kidney cancer: No specific subtypes
 - Liver Cancer: No specific subtypes
 - Lung cancer: No specific subtypes
 - Ovarian cancer: No specific subtypes
 - Pancreatic cancer: No specific subtypes
 - Prostate cancer: No specific subtypes
 - Skin cancer: No specific subtypes
 - Stomach cancer: No specific subtypes
 - Thyroid cancer: No specific subtypes
 - Uterine cancer: Carcinosarcoma (malignant mixed Müllerian / mixed mesodermal tumor), Endometrioid adenocarcinoma, 
 High-grade endometrial stromal sarcoma (HG-ESS), Inflammatory myofibroblastic tumor (IMT), 
 Low-grade endometrial stromal sarcoma (LG-ESS), Müllerian adenosarcoma (MAS), NTRK-rearranged spindle-cell sarcoma, 
 Perivascular epithelioid cell tumor (PEComa), Rhabdomyosarcoma (RMS), SMARCA4-deficient uterine sarcoma (SDUS), 
 Undifferentiated / dedifferentiated carcinoma, Undifferentiated uterine sarcoma (UUS), Uterine clear-cell carcinoma, 
 Uterine leiomyosarcoma (uLMS), Uterine serous carcinoma, Uterine tumor resembling ovarian sex-cord tumor (UTROSCT)"""