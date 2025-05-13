import datetime
import io
import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Union

import PyPDF2
import fitz  # PyMuPDF
import numpy as np
import openai
import pdfplumber
import pytesseract
import tiktoken
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm

# =============================================================================
# Configuration and Setup
# =============================================================================

# Configure logging
log_dir = Path("logs")                                                  # Define path for storing log files
log_dir.mkdir(exist_ok=True)                                            # Create logs directory if it doesn't exist

# Set up file handler to save logs
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # Get current timestamp for log filename
log_file = log_dir / f"pathology_analyzer_{current_time}.log"           # Create log file path with timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,                                                 # Set overall logging level to INFO
    format="%(asctime)s | %(levelname)s | %(message)s",                 # Define log message format with timestamp
    datefmt="%Y-%m-%d %H:%M:%S",                                        # Define date format for log entries
    handlers=[
        logging.StreamHandler(),                                        # Log to console
        logging.FileHandler(log_file)                                   # Log to file
    ]
)

# Configure 3rd party loggers
logging.getLogger("openai").setLevel(logging.WARNING)                   # Reduce verbosity of OpenAI's logger
logging.getLogger("httpx").setLevel(logging.WARNING)                    # Reduce verbosity of httpx (used by OpenAI)

logger = logging.getLogger(__name__)                                    # Create logger for this module
logger.info(f"Starting Pathology Report Analyzer. Logs will be saved to {log_file}")

# Directory and file paths
ROOT_DIR = Path("cases")                                                # Root directory containing case folders
NCCN_PDF_PATH = Path("uterine_core.pdf")                                # Path to NCCN guidelines PDF
EMBEDDINGS_CACHE_PATH = Path("nccn_embeddings_cache.pkl")               # Path for saved embeddings

# Regular expression to extract JSON from text
json_block = re.compile(r'{.*}', re.S)

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
 - Uterine cancer: Carcinosarcoma (malignant mixed Müllerian / mixed mesodermal tumor), Endometrioid adenocarcinoma, High-grade endometrial stromal sarcoma (HG-ESS), Inflammatory myofibroblastic tumor (IMT), Low-grade endometrial stromal sarcoma (LG-ESS), Müllerian adenosarcoma (MAS), NTRK-rearranged spindle-cell sarcoma, Perivascular epithelioid cell tumor (PEComa), Rhabdomyosarcoma (RMS), SMARCA4-deficient uterine sarcoma (SDUS), Undifferentiated / dedifferentiated carcinoma, Undifferentiated uterine sarcoma (UUS), Uterine clear-cell carcinoma, Uterine leiomyosarcoma (uLMS), Uterine serous carcinoma, Uterine tumor resembling ovarian sex-cord tumor (UTROSCT)"""

# List of cancer organ types
ORGAN_TYPES_STR = (
    "Bladder cancer, Blood/Lymph Cancer, Brain cancer, Breast cancer, Cervical cancer, "
    "Colorectal cancer, Esophageal cancer, Kidney cancer, Liver Cancer, Lung cancer, "
    "Ovarian cancer, Pancreatic cancer, Prostate cancer, Skin cancer, Stomach cancer, "
    "Thyroid cancer, Uterine cancer"
)


# =============================================================================
# Exception Classes
# =============================================================================

class PathologyAnalyzerError(Exception):
    """Base exception for all pathology analyzer errors.

    Serves as the root exception class for the Pathology Analyzer application.
    All other exception types in this application inherit from this class to
    provide a consistent error hierarchy and to allow catching all application-specific
    errors with a single exception type.
    """
    pass

class TextExtractionError(PathologyAnalyzerError):
    """Raised when text cannot be extracted from a PDF document.

    This exception is raised when all PDF text extraction methods (PyMuPDF, pdfplumber,
    PyPDF2, and OCR) have failed to extract readable text from a PDF document.
    It typically indicates either a corrupt PDF file, a scanned document with poor
    image quality, or a PDF with security settings that prevent text extraction.
    """
    pass

class OpenAIAPIError(PathologyAnalyzerError):
    """Raised when there's an error with the OpenAI API communication.

    This exception is raised when a request to the OpenAI API fails, times out,
    or returns an unexpected response. This could be due to authentication issues,
    rate limiting, service outages, malformed requests, or other API-related problems.
    The exception typically includes the original error message from the OpenAI client.
    """
    pass

class EmbeddingError(PathologyAnalyzerError):
    """Raised when there's an error generating text embeddings.

    This exception is raised when the system fails to create vector embeddings
    for text using OpenAI's embedding API. This could be due to API errors,
    token limit exceedance, or other issues with the embedding process.
    The exception includes details about the specific embedding operation that failed.
    """
    pass

class VerificationError(PathologyAnalyzerError):
    """Raised when verification of extracted data fails.

    This exception is raised during the verification stage when the system
    determines that the extracted information from a pathology report is
    potentially incorrect, incomplete, or inconsistent with the source document.
    It may include details about which specific fields failed verification and why.
    """
    pass


# =============================================================================
# Configuration Class
# =============================================================================

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
    """
    # API Key and Models
    openai_api_key: str = os.getenv("OPENAI_API_KEY")                   # Get API key from environment variable
    openai_model: str = "o4-mini"                                       # Default model for main analysis
    openai_embed_model: str = "text-embedding-3-large"                  # Default model for embeddings
    openai_repair_model: str = "gpt-3.5-turbo"                          # Default model for JSON repair

    # Paths
    root_dir: Path = Path("cases")                                      # Path to case directories
    nccn_pdf_path: Path = Path("uterine_core.pdf")                      # Path to guidelines PDF
    embeddings_cache_path: Path = Path("nccn_embeddings_cache.pkl")     # Path for cache
    log_dir: Path = Path("logs")                                        # Path for log files

    # Extraction parameters
    chunk_token_size: int = 500                                         # Size of chunks for tokenization
    k_guideline_chunks: int = 8                                         # Number of chunks to retrieve
    sim_threshold: float = 0.25                                         # Similarity threshold for relevance
    max_total_tokens: int = 110_000                                     # Max tokens for API calls

    # Verification parameters
    verification_enabled: bool = True                                   # Enable/disable verification
    verification_threshold: float = 0.8                                 # Confidence threshold
    detailed_verification: bool = True                                  # Enable detailed verification

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
        return self


# =============================================================================
# Text Processing Utilities
# =============================================================================

# Initialize tokenizer once for reuse
_enc = tiktoken.get_encoding("cl100k_base")   # This is NOT future-proof. But currently correct


def num_tokens(text: str) -> int:
    """Count the number of tokens in a string using the cl100k_base tokenizer.

    This function counts how many tokens a given text string would use when processed
    by OpenAI models. This is important for staying within token limits for API calls.
    Uses the cl100k_base tokenizer which is compatible with many OpenAI models.

    Args:
        text (str): The text string to tokenize

    Returns:
        int: The number of tokens in the text
    """
    return len(_enc.encode(text))

def split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks of maximum token size.

    This function divides a long text string into smaller chunks, each containing
    at most max_tokens tokens. This is useful for processing long documents while
    staying within the token limits of AI models.

    The function first tokenizes the entire text, then divides the tokens into
    chunks of the specified size, and finally decodes each chunk back to text.
    This ensures that the splits occur at token boundaries rather than character
    boundaries, which is important for accurate token counting.

    Args:
        text (str): The text to split
        max_tokens (int): Maximum number of tokens per chunk

    Returns:
        List[str]: List of text chunks, each containing at most max_tokens tokens
    """
    tokens = _enc.encode(text)                                                      # Encode the text into tokens
    chunks = [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)] # Split into chunks
    return [_enc.decode(chunk) for chunk in chunks]                                 # Decode each chunk back to text

def safe_json(text: Union[str, list, dict]) -> dict:
    """Parse and repair JSON from model output text.

    This function attempts to convert various input types into a valid JSON dictionary.
    It handles already-parsed dictionaries, lists of dictionaries, and strings that contain
    JSON. If standard parsing fails, it uses multiple fallback mechanisms including regex
    extraction and even an LLM to repair malformed JSON.

    This is particularly useful for handling LLM outputs that might contain JSON but with
    slight formatting errors or extra text surrounding the JSON object.

    Args:
        text (Union[str, list, dict]): The input to parse into a JSON dictionary.
            Can be an already-parsed dictionary, a list of dictionaries, or a string
            containing JSON (possibly with errors or extra text)

    Returns:
        dict: A dictionary parsed from the input. If all parsing attempts fail,
              returns an error dictionary with partial content from the input.

    Flow:
        1. If input is already a dict, return it directly
        2. If input is a list of dicts, return the first dict
        3. If input is a string, attempt direct JSON parsing
        4. If that fails, try to extract JSON using regex
        5. If that fails, use OpenAI to repair the JSON
        6. If all attempts fail, return an error dictionary
    """
    # If it's already a dict, return it
    if isinstance(text, dict):
        return text

    # If it's a list, try to convert it to a dictionary
    if isinstance(text, list):
        if all(isinstance(item, dict) for item in text):    # Check if all items are dictionaries
            return text[0]                                  # Return the first dict in the list
        text = json.dumps(text)                             # Otherwise convert list to JSON string

    # Ensure we're working with a string
    if not isinstance(text, str):                           # Check if input is not a string
        text = str(text)                                    # Convert to string if needed

    # First attempt: direct JSON parsing
    try:
        return json.loads(text)                             # Attempt to parse as JSON directly
    except json.JSONDecodeError:
        # Second attempt: find JSON block using regex
        match = json_block.search(text)                     # Search for JSON-like pattern
        if match:
            try:
                return json.loads(match.group(0))           # Try to parse the matched text
            except json.JSONDecodeError:
                pass

    # Final attempt: use a model to repair the JSON (vibe-coding, no other choice)
    try:
        repair_prompt = (
                "The following is meant to be a JSON object but is invalid.\n"
                "Fix syntax ONLY and return valid JSON with the same keys/values.\n\n"
                "-----\n" + text[:5000] + "\n-----"  # Limit to 5000 chars to avoid token limits
        )

        repair_resp = openai.chat.completions.create(               # Call OpenAI API for repair
            model="gpt-3.5-turbo",                                  # Use GPT-3.5-turbo model for repair
            messages=[{"role": "user", "content": repair_prompt}]   # Send the repair prompt
        )
        repaired = repair_resp.choices[0].message.content           # Extract repaired JSON from response
        return json.loads(repaired)                                 # Parse the repaired JSON
    except Exception as e:
        logger.error(f"JSON repair failed: {e}")                    # Log the error
        return {
            "error": "Could not parse JSON response",
            "partial_content": text[:200] + "..." if len(text) > 200 else text
        }

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Computes the cosine of the angle between two vectors, which is a measure of
    similarity between 1 (identical direction) and -1 (opposite direction),
    with 0 indicating orthogonality (no similarity).

    This implementation uses the dot product and vector norms, with a small epsilon
    value (1e-10) added to the denominator to prevent division by zero.

    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector

    Returns:
        float: Cosine similarity between vectors a and b, in range [-1, 1]
    """
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# =============================================================================
# PDF Processing Module
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((IOError, ValueError))
)
def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF with multiple fallback methods.

    This function attempts to extract text from a PDF file using a series of increasingly
    robust methods, from fastest to slowest. It tries each method in sequence until
    one succeeds in extracting meaningful text (defined as at least 100 characters).

    Methods attempted, in order:
    1. PyMuPDF (fastest, good general-purpose extraction)
    2. pdfplumber (better layout preservation)
    3. PyPDF2 (different parsing engine as fallback)
    4. OCR using pytesseract (slowest but works on scanned documents)

    The function is decorated with @retry to automatically retry on IOError or ValueError
    up to 3 times with exponential backoff between attempts.

    Args:
        pdf_path (Path): Path to the PDF file to extract text from

    Returns:
        str: Extracted text from the PDF

    Raises:
        TextExtractionError: If all extraction methods fail
    """
    logger.info(f"Extracting text from {pdf_path.name}")

    # Method 1: PyMuPDF (fastest)
    try:
        with fitz.open(pdf_path) as doc:                                    # Open PDF with PyMuPDF
            text = "\n".join(page.get_text("text") for page in doc)         # Extract text from each page
            if len(text.strip()) > 100:                                     # Basic quality check
                logger.info(f"Successfully extracted text using PyMuPDF")   # Log success
                return text                                                 # Return the extracted text
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")

    # Method 2: pdfplumber (better layout preservation)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:                         # Open PDF with pdfplumber
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)     # Extract text from each page
            if len(text.strip()) > 100:                                     # Check if extracted text is meaningful
                logger.info(f"Successfully extracted text using pdfplumber")# Log success
                return text                                                 # Return the extracted text
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")

    # Method 3: PyPDF2 (fallback)
    try:
        with open(pdf_path, 'rb') as f:                                             # Open PDF in binary mode
            reader = PyPDF2.PdfReader(f)                                            # Create PDF reader
            text = "\n".join(page.extract_text() or "" for page in reader.pages)    # Extract text from each page
            if len(text.strip()) > 100:                                             # Check if extracted text is meaningful
                logger.info(f"Successfully extracted text using PyPDF2")            # Log success
                return text                                                         # Return the extracted text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")

    # Method 4: OCR each page image (slowest but most reliable)
    logger.info("Falling back to OCR... this may take several minutes")
    text_pages = []                                                                 # Initialize list to store OCR results
    try:
        with fitz.open(pdf_path) as doc:                                            # Open PDF with PyMuPDF
            for page in doc:                                                        # Process each page
                pix = page.get_pixmap(dpi=400)                                      # Convert page to image at 400 DPI
                img = Image.open(io.BytesIO(pix.tobytes()))                         # Convert to PIL Image
                text_pages.append(pytesseract.image_to_string(img))                 # Perform OCR on the image

        text = "\n".join(text_pages)                                                # Join OCR results from all pages
        if len(text.strip()) > 100:                                                 # Check if OCR text is meaningful
            logger.info(f"Successfully extracted text using OCR")
            return text                                                             # Return the OCR text
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")

    # If all methods fail
    raise TextExtractionError(f"Could not extract meaningful text from {pdf_path.name} using any method")


# =============================================================================
# Embedding and Retrieval Functions
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
@lru_cache(maxsize=1024)  # Cache embeddings to avoid re-computing
def embed(text: str) -> np.ndarray:
    """Create an embedding vector for text using OpenAI's embedding model.

    This function converts a text string into a vector representation (embedding)
    using OpenAI's text-embedding-3-large model. The resulting embedding can be used
    for semantic similarity comparisons and information retrieval.

    The function is decorated with:
    - @retry: Automatically retries failed API calls up to 3 times with exponential backoff
    - @lru_cache: Caches results to avoid redundant API calls for the same text

    Args:
        text (str): The text to create an embedding for

    Returns:
        np.ndarray: A numpy array containing the embedding vector

    Raises:
        EmbeddingError: If the embedding creation fails after retries
    """
    try:
        return np.array(                            # Convert response to numpy array
            openai.embeddings.create(               # Call OpenAI embeddings API
                model="text-embedding-3-large",     # Specify embedding model
                input=text                          # Provide text to embed
            ).data[0].embedding                     # Extract the embedding vector
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise EmbeddingError(f"Failed to create embedding: {e}")

def load_or_create_embeddings(config: AnalyzerConfig):
    """Load NCCN guideline embeddings from cache if available, otherwise create and save them.

    This function efficiently manages embeddings for NCCN guidelines by:
    1. First attempting to load pre-computed embeddings from a cache file
    2. If that fails, extracting text from the NCCN guidelines PDF
    3. Chunking the extracted text into manageable pieces
    4. Computing embeddings for each chunk
    5. Saving the chunks and embeddings to cache for future use

    This approach optimizes performance by avoiding redundant computation of embeddings
    across multiple runs of the application.

    Args:
        config (AnalyzerConfig): Configuration object containing paths and parameters

    Returns:
        tuple: A tuple containing:
            - guide_chunks (List[str]): List of text chunks from the NCCN guidelines
            - guide_embeds (np.ndarray): Matrix of embeddings for each chunk

    Raises:
        Exception: Propagates any exceptions from text extraction or embedding generation
    """

    # Check if embeddings cache exists
    if config.embeddings_cache_path.exists():
        logger.info(f"Loading NCCN embeddings from cache: {config.embeddings_cache_path}")
        try:
            with open(config.embeddings_cache_path, 'rb') as f:
                cache_data = pickle.load(f)                                     # Load cached data
                guide_chunks = cache_data['chunks']                             # Extract chunks from cache
                guide_embeds = cache_data['embeds']                             # Extract embeddings from cache
                logger.info(f"Loaded {len(guide_chunks)} chunks from cache")
                return guide_chunks, guide_embeds
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}. Regenerating...")

    # If cache doesn't exist or loading failed, generate embeddings
    logger.info("Loading NCCN guideline PDF...")
    try:
        guideline_raw = extract_text(config.nccn_pdf_path)                      # Extract text from NCCN PDF
        # Split guidelines into manageable chunks
        guide_chunks = split_by_tokens(guideline_raw, config.chunk_token_size)  # Split into chunks
        logger.info(f"Chunked NCCN guideline into {len(guide_chunks)} pieces ≤ {config.chunk_token_size} tokens each")

        # Embed all chunks for later similarity search
        logger.info("Embedding guideline chunks... this runs once and will be cached")
        guide_embeds = np.vstack([                                              # Stack embeddings into a matrix
            embed(chunk) for chunk in tqdm(guide_chunks, desc="Embed NCCN")
        ])

        # Save embeddings to cache
        logger.info(f"Saving embeddings to cache: {config.embeddings_cache_path}")
        with open(config.embeddings_cache_path, 'wb') as f:
            pickle.dump({'chunks': guide_chunks, 'embeds': guide_embeds}, f)

        return guide_chunks, guide_embeds
    except Exception as e:
        logger.error(f"Failed to process NCCN guidelines: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def top_k_guideline(query: str, guide_chunks: List[str], guide_embeds: np.ndarray,
                    k: int = 8, threshold: float = 0.25) -> List[str]:
    """Find the top-k most relevant NCCN guideline chunks for a query.

    This function performs semantic search to identify the most relevant
    segments of the NCCN guidelines for a given pathology report text.
    It works by:
    1. Converting the query text to an embedding vector
    2. Computing similarity scores between the query embedding and all guideline embeddings
    3. Selecting the top k chunks that exceed the similarity threshold

    The function is decorated with @retry to automatically retry on failures.

    Args:
        query (str): The text to find relevant guidelines for (typically a pathology report)
        guide_chunks (List[str]): List of text chunks from the NCCN guidelines
        guide_embeds (np.ndarray): Matrix of embeddings for each guideline chunk
        k (int, optional): Maximum number of chunks to return. Defaults to 8.
        threshold (float, optional): Minimum similarity score to include a chunk. Defaults to 0.25.

    Returns:
        List[str]: List of the most relevant guideline text chunks, ordered by relevance
    """
    # Embed the query
    q_emb = embed(query)

    # Calculate similarity to all guideline chunks
    sims = np.array([cosine(e, q_emb) for e in guide_embeds])

    # Get indices of top k chunks above threshold
    idx = [i for i in sims.argsort()[::-1] if sims[i] >= threshold][:k]

    # Return the corresponding chunks
    return [guide_chunks[i] for i in idx]


# =============================================================================
# OpenAI API Interaction
# =============================================================================

INSTRUCTIONS_BLOCK = """
        For the given pathology report, extract the following information with great attention to detail:

        1. Cancer Organ Type: Identify the specific organ affected from our database list above.
           - Choose EXACTLY ONE of these organ types: {organ_types}
           - If the organ is not in our list, choose the closest match
           - If truly not determinable, indicate "Not specified"

        2. Cancer Subtype: Identify the specific subtype of cancer within that organ.
           - Choose from the corresponding subtypes for the organ you selected
           - If the subtype is not in our list for that organ, provide your best medical assessment
           - Be as specific as the report allows

        3. FIGO Stage: Extract the FIGO staging information if present. Look for:
           - Any mentions of "FIGO" followed by stage numbers/letters (e.g., "FIGO Stage IIIC2")
           - Words like "stage" followed by Roman numerals (I, II, III, IV) with possible subdivisions (A, B, C)
           - Explicit statements about depth of invasion, myometrial invasion, or serosal involvement
           - Descriptions like "confined to endometrium" (typically Stage IA) or "invading outer half of myometrium" (typically Stage IB)
           - If grade information is provided alongside stage, include it

        4. Final Pathologic Stage: Extract the final pathologic stage information, including any TNM staging. Look for:
           - TNM notation like "pT1a", "pT2", "pN0", "pN1", "pM0", etc.
           - Tumor size measurements with T-classification (e.g., "T1: tumor ≤ 2cm")
           - Lymph node status (e.g., "lymph nodes negative 0/12" would be N0)
           - Statements about metastasis or distant spread
           - Look for staging summary in "DIAGNOSIS" or "FINAL DIAGNOSIS" sections

        5. Recommended Treatment: Based STRICTLY on NCCN guidelines for the identified cancer type and stage. Your recommendations MUST:
           - Be derived EXCLUSIVELY from current NCCN guidelines, not from general knowledge
           - Match the specific cancer type, stage, and any additional factors (like grade, receptor status) identified in the report
           - Include first-line treatment options in a concise format (e.g., "Surgery followed by adjuvant chemotherapy")
           - Mention if there are multiple standard-of-care options per NCCN guidelines
           - Be as specific as possible given the information available
           - If there's not enough information to determine a treatment recommendation, state: "Insufficient information to determine NCCN-based treatment recommendation"
           - DO NOT make recommendations that are not explicitly part of NCCN guidelines

        6. Description: Write a brief 1-2 sentence professional summary of the document.
           - Use appropriate medical terminology
           - Be specific about findings
           - Include key diagnostic information

        7. Patient Notes: Write 2-4 patient-friendly sentences explaining this document.
           - Use simple, non-technical language
           - Explain medical terms when necessary
           - Be compassionate but factual
           - Explain what the findings mean for the patient
           - Avoid being alarming while still being truthful
           - Include a mention of the recommended treatment approach

        IMPORTANT: Examine the document VERY CAREFULLY for staging information. Even if staging is not labeled explicitly, look for clinical descriptions that indicate staging according to NCCN guidelines."""

SYSTEM_MSG = (
    "You are a medical AI assistant specializing in oncology and pathology reports. "
    "Follow NCCN guidelines STRICTLY. Do all reasoning silently and output ONLY the JSON object asked for."
)

PROMPT_TEMPLATE = """Our database contains the following organ types: {organ_types}

And these subtypes for each organ: {subtypes}

<<<PATHOLOGY REPORT>>>
{report}
<<<END REPORT>>>

<<<RELEVANT NCCN GUIDELINE EXCERPTS>>>
{nccn}
<<<END NCCN>>>

{instructions}
"""

VERIFICATION_SYSTEM_MSG = (
    "You are an expert medical data verification assistant. Your role is to critically evaluate "
    "extracted data from pathology reports for accuracy, completeness, and consistency. "
    "Be thorough and precise in your verification process."
)

VERIFICATION_PROMPT = """
I need you to verify the accuracy of information extracted from a pathology report. 

PATHOLOGY REPORT:
```
{report}
```

EXTRACTED INFORMATION:
```
{extracted_info}
```

RELEVANT NCCN GUIDELINES:
```
{nccn}
```

Please carefully review and check:
1. Are all fields correctly extracted based on the report?
2. Is the Cancer Organ Type correct and consistent with the report?
3. Is the Cancer Subtype correct for the identified organ?
4. Is the FIGO stage accurately captured?
5. Is the Pathologic Stage accurate?
6. Is the treatment recommendation consistent with NCCN guidelines for this specific cancer type and stage?
7. Is the professional description accurate and complete?
8. Are the patient notes clear, compassionate, and accurate?

Provide verification results in the following JSON format:
{{
    "verification_result": "PASS" or "FAIL",
    "confidence_score": (0.0 to 1.0 indicating overall confidence),
    "field_verification": {{
        "cancer_organ_type": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }},
        "cancer_subtype": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }},
        "figo_stage": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }},
        "pathologic_stage": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }},
        "recommended_treatment": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }},
        "description": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }},
        "patient_notes": {{ "correct": true/false, "confidence": 0.0-1.0, "issues": "description of any issues" }}
    }},
    "missing_information": [list of information that should have been extracted but wasn't],
    "incorrect_fields": [list of fields with incorrect information],
    "recommended_corrections": {{
        "field_name": "corrected value",
        ...
    }},
    "overall_assessment": "brief assessment of extraction quality"
}}
"""


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((IOError, ValueError))
)
def call_openai(msgs: List[Dict]) -> Dict:
    """Send a request to OpenAI API and parse the response as JSON.

    This function sends a conversation to the OpenAI Chat Completions API
    and processes the response into a structured JSON format. It handles
    errors and retry logic for robustness against transient API issues.

    The function is configured to retry up to 4 times with exponential backoff
    when encountering IOError or ValueError exceptions.

    Args:
        msgs (List[Dict]): List of message dictionaries in the OpenAI Chat format,
                          each containing at minimum 'role' and 'content' keys

    Returns:
        Dict: The parsed JSON response from the OpenAI API

    Raises:
        OpenAIAPIError: If the API call fails after retries or returns unparseable content
    """
    try:
        response = openai.chat.completions.create(
            model="o4-mini",                        # Use o4-mini model
            messages=msgs                           # Pass in the messages
        )
        return safe_json(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise OpenAIAPIError(f"Error calling OpenAI API: {e}")


def analyse_report(
        report_text: str,
        guide_chunks: List[str],
        guide_embeds: np.ndarray,
        config: AnalyzerConfig) -> Dict:
    """
    Analyze a pathology report using OpenAI API and relevant NCCN guidelines.

    This function implements a comprehensive two-stage approach to extract and verify
    structured information from pathology reports:

    Stage 1: Initial Extraction
    - Identifies relevant NCCN guideline chunks for the report using semantic search
    - Constructs a prompt combining the report, guidelines, and extraction instructions
    - Calls the OpenAI API to extract structured information including:
      * Cancer organ type and subtype
      * FIGO staging information
      * Pathologic staging
      * Treatment recommendations based on NCCN guidelines
      * Professional medical description
      * Patient-friendly notes

    Stage 2: Verification and Correction (if enabled)
    - Takes the extracted information and sends it to a verification step
    - Verifies accuracy, completeness, and consistency of the extraction
    - Identifies and corrects any issues in the extracted data
    - Adds verification metadata to the final result

    Args:
        report_text (str): The text of the pathology report to analyze
        guide_chunks (List[str]): List of text chunks from the NCCN guidelines
        guide_embeds (np.ndarray): Matrix of embeddings for each guideline chunk
        config (AnalyzerConfig): Configuration parameters for the analysis

    Returns:
        Dict: A dictionary containing the structured information extracted from the report,
              including verification results if verification was enabled
    """
    # Find relevant NCCN guideline chunks for this report
    nccn_text = "\n\n".join(top_k_guideline(                            # Get and join relevant guideline chunks
        report_text, guide_chunks, guide_embeds,                        # Pass report and guideline data
        k=config.k_guideline_chunks, threshold=config.sim_threshold     # Use config parameters
    ))

    # Log the analysis process
    logger.info(f"Analyzing report with {len(report_text)} chars and {len(nccn_text)} chars of NCCN guidelines")

    # Create the full prompt with report and guidelines
    user_msg = PROMPT_TEMPLATE.format(
        organ_types=ORGAN_TYPES_STR,        # Include organ types list
        subtypes=SUBTYPES_STR,              # Include subtypes list
        report=report_text,                 # Include report text
        nccn=nccn_text,                     # Include relevant NCCN guidelines
        instructions=INSTRUCTIONS_BLOCK     # Include extraction instructions
    )

    # Stage 1: Initial extraction
    logger.info("Performing initial extraction...")
    try:
        primary_extraction = call_openai([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg}
        ])

        logger.info(f"Initial extraction successful: {list(primary_extraction.keys())}")

        # Stage 2: Verification and correction (only if enabled and stage 1 succeeds)
        if config.verification_enabled:
            logger.info("Performing verification...")

            # Create verification prompt
            verification_prompt = VERIFICATION_PROMPT.format(
                report=report_text[:5000],                                  # Limit length to avoid token issues
                extracted_info=json.dumps(primary_extraction, indent=2),    # Include JSON of extraction
                nccn=nccn_text[:5000]                                       # Limit length to avoid token issues
            )

            # Call verification API
            try:
                verification_result = call_openai([                         # Call OpenAI API for verification
                    {"role": "system", "content": VERIFICATION_SYSTEM_MSG}, # Include verification system message
                    {"role": "user", "content": verification_prompt}        # Include verification prompt
                ])

                # Process verification results
                if 'verification_result' in verification_result:
                    if verification_result['verification_result'] == 'PASS':# Check if verification passed
                        logger.info("Verification PASSED")
                        return {
                            **primary_extraction,                           # Include all primary extraction fields
                            'verification': {
                                'passed': True,
                                'confidence': verification_result.get('confidence_score', 1.0),
                                'assessment': verification_result.get('overall_assessment',
                                                                      'Extraction verified successfully')
                            }
                        }
                    else:
                        logger.info("\033[91mVerification FAILED\033[0m - applying corrections")

                        # Apply corrections to the primary extraction
                        corrected_extraction = dict(primary_extraction)
                        if 'recommended_corrections' in verification_result:
                            for field, value in verification_result['recommended_corrections'].items():
                                if field in corrected_extraction:
                                    corrected_extraction[field] = value

                        # Add verification metadata
                        corrected_extraction['verification'] = {
                            'passed': False,                                                            # Mark as failed
                            'confidence': verification_result.get('confidence_score', 0.0),             # Include confidence
                            'assessment': verification_result.get('overall_assessment',
                                                                  'Extraction required corrections'),   # Include assessment
                            'field_issues': verification_result.get('field_verification', {}),          # Include field issues
                            'incorrect_fields': verification_result.get('incorrect_fields', [])         # Include incorrect fields
                        }

                        return corrected_extraction
                else:
                    logger.warning("Verification returned unexpected format")
                    # In case of unexpected format, return primary with a warning
                    return {
                        **primary_extraction,
                        'verification': {                                               # Add verification metadata
                            'passed': False,                                            # Mark as failed
                            'confidence': 0.5,                                          # Set medium confidence
                            'assessment': 'Verification produced unexpected results'    # Include assessment
                        }
                    }
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                # If verification fails, still return primary results with a warning
                return {
                    **primary_extraction,                                       # Include all primary extraction fields
                    'verification': {                                           # Add verification metadata
                        'passed': False,                                        # Mark as failed
                        'confidence': 0.0,                                      # Set zero confidence
                        'assessment': f'Verification process failed: {str(e)}'
                    }
                }

        # If verification is disabled, return the primary extraction
        return primary_extraction

    except Exception as e:
        # If primary extraction fails, construct minimal results with error information
        logger.error(f"Primary extraction failed: {e}")
        return {
            "error": f"Extraction failed: {str(e)}",
            "cancer_organ_type": "Not specified",
            "cancer_subtype": "Not specified",
            "figo_stage": "Not specified",
            "pathologic_stage": "Not specified",
            "recommended_treatment": "Not specified",
            "description": "Failed to analyze pathology report due to technical error.",
            "patient_notes": "There was a technical issue processing your report. Please consult with your healthcare provider for interpretation.",
            "verification": {
                "passed": False,
                "confidence": 0.0,
                "assessment": f"Extraction failed: {str(e)}"
            }
        }


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_all_folders(config: AnalyzerConfig) -> None:
    """
    Process all case folders, extract information from PDFs, and save analysis.

    This function is the main processing pipeline for batch analysis of pathology reports:
    1. Loads or creates embeddings for NCCN guidelines
    2. Identifies all case folders in the root directory
    3. For each case folder:
       - Checks if it has already been processed
       - Finds PDF files containing pathology reports
       - Extracts text from the PDF
       - Analyzes the report text with relevant NCCN guidelines
       - Saves the analysis results to a JSON file in the case folder

    The function handles error conditions and logs progress throughout the process.
    It skips folders that have already been processed (have an analysis.json file)
    and continues with the next folder in case of errors.

    Args:
        config (AnalyzerConfig): Configuration object containing paths and parameters

    Returns:
        None: Results are saved to disk rather than returned
    """
    # Load NCCN guidelines embeddings
    guide_chunks, guide_embeds = load_or_create_embeddings(config)

    # Find all directories that don't start with '.'
    cases = [p for p in config.root_dir.iterdir()
             if p.is_dir() and not p.name.startswith('.')]

    # Process each case folder
    for case in tqdm(sorted(cases), desc="cases"):
        out_file = case / "analysis.json"

        # Skip if already processed
        if out_file.exists():
            logger.info(f"Skip {case.name} (already processed)")
            continue

        # Find PDF files in the case folder
        pdfs = sorted(case.glob("*.pdf"))

        # Validate PDF existence
        if not pdfs:
            logger.warning(f"No PDF found in {case}")
            continue

        # Warn if multiple PDFs found
        if len(pdfs) > 1:
            logger.warning(f"{case}: multiple PDFs, using {pdfs[0].name}")

        try:
            # Extract text from the PDF
            text = extract_text(pdfs[0])

            # Analyze the report
            analysis = analyse_report(text, guide_chunks, guide_embeds, config)

            # Save results to JSON file
            out_file.write_text(json.dumps(analysis, indent=2))
            logger.info(f"Wrote analysis to {out_file.relative_to(config.root_dir)}")

        except Exception as e:
            logger.error(f"{case.name}: {e}")


# =============================================================================
# Complete "Single Process" Multi-agent Verification System
# =============================================================================

class VerificationAgent:
    """
    Enhanced verification agent for pathology report data.

    This class provides detailed verification and cross-checking of extracted data
    from pathology reports. It acts as a separate validation layer to ensure the
    accuracy, completeness, and consistency of extracted information.

    The verification agent can perform both general verification of all extracted fields
    and detailed verification of critical fields, with the ability to apply corrections
    to the extracted data based on verification results.

    Key capabilities:
    - Verify extracted information against the original report and NCCN guidelines
    - Provide confidence scores for each field and overall extraction
    - Identify missing or incorrect information
    - Suggest corrections for inaccurate fields
    - Perform detailed field-by-field verification for critical information

    This implements a multi-agent verification system, where one model performs the
    initial extraction and another performs independent verification as a quality
    control measure.
    """

    def __init__(self, config: AnalyzerConfig):
        """Initialize the verification agent.

        Sets up the verification agent with the provided configuration parameters.

        Args:
            config (AnalyzerConfig): Configuration object containing verification settings,
                                     including model selection and verification thresholds
        """
        self.config = config
        self.openai_model = config.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def verify(self, report_text: str, extraction: Dict, nccn_text: str) -> Dict:
        """
        Verify the extraction results against the report and guidelines.

        This method performs comprehensive verification of extracted pathology report data
        by comparing it to the original report text and NCCN guidelines. It sends all
        information to an LLM with instructions to verify accuracy, completeness, and
        consistency across all extracted fields.

        The method is decorated with @retry to handle transient API failures with
        automatic retries and exponential backoff.

        Args:
            report_text (str): Original report text from the PDF
            extraction (Dict): Extracted structured information to verify
            nccn_text (str): Relevant NCCN guidelines text for reference

        Returns:
            Dict: A dictionary containing verification results with the following keys:
                - verification_result: "PASS" or "FAIL"
                - confidence_score: Overall confidence (0.0-1.0)
                - field_verification: Per-field verification details
                - missing_information: Any information missing from extraction
                - incorrect_fields: List of fields with issues
                - recommended_corrections: Suggested fixes for incorrect fields
                - overall_assessment: Summary assessment of extraction quality
                - timestamp: When verification was performed
                - model_used: The model used for verification
        """
        try:
            # Create verification prompt
            verification_prompt = VERIFICATION_PROMPT.format(
                report=report_text[:5000],                                  # Limit length to avoid token issues
                extracted_info=json.dumps(extraction, indent=2),            # Include JSON of extraction
                nccn=nccn_text[:5000]                                       # Limit length to avoid token issues
            )

            # Call OpenAI API for verification
            response = openai.chat.completions.create(
                model=self.openai_model,                                    # Use configured model
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM_MSG}, # Include system message
                    {"role": "user", "content": verification_prompt}        # Include verification prompt
                ]
            )

            # Parse the result
            result_text = response.choices[0].message.content                       # Extract response text
            verification_result = safe_json(result_text)                            # Parse JSON from response

            # Enhance with verification metadata
            verification_result['timestamp'] = datetime.datetime.now().isoformat()  # Add timestamp
            verification_result['model_used'] = self.openai_model                   # Add model info

            return verification_result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "verification_result": "ERROR",
                "error": str(e),
                "confidence_score": 0.0,
                "timestamp": datetime.datetime.now().isoformat(),
                "overall_assessment": "Verification process failed due to technical error"
            }

    def apply_corrections(self, extraction: Dict, verification: Dict) -> Dict:
        """
        Apply corrections from verification to the extracted data.

        This method takes the original extracted data and the verification results,
        and creates a corrected version of the extraction by applying any recommended
        corrections from the verification process. It also adds detailed verification
        metadata to the result.

        Args:
            extraction (Dict): Original extraction data before corrections
            verification (Dict): Verification results containing recommended corrections
                               and verification metadata

        Returns:
            Dict: A new dictionary containing the corrected extraction data with all
                  original fields plus verification metadata, including:
                  - passed: Boolean indicating if verification passed
                  - confidence: Overall confidence score (0.0-1.0)
                  - assessment: Text description of verification results
                  - field_issues: Detailed issues by field
                  - incorrect_fields: List of field names with issues
                  - timestamp: When verification was performed
        """
        corrected = dict(extraction)

        # Apply direct corrections if available
        if 'recommended_corrections' in verification:
            for field, value in verification['recommended_corrections'].items():
                if field in corrected:
                    corrected[field] = value

        # Add verification metadata
        corrected['verification'] = {
            'passed': verification.get('verification_result') == 'PASS',
            'confidence': verification.get('confidence_score', 0.0),
            'assessment': verification.get('overall_assessment', 'Extraction verification completed'),
            'field_issues': verification.get('field_verification', {}),
            'incorrect_fields': verification.get('incorrect_fields', []),
            'timestamp': verification.get('timestamp', datetime.datetime.now().isoformat())
        }

        return corrected

    def detailed_field_verification(self, report_text: str, extraction: Dict) -> Dict:
        """
        Perform detailed verification of individual fields.

        This method provides an additional layer of verification by independently
        checking each critical field against the original report. It's particularly
        useful for high-stakes information like cancer type, subtype, and staging.

        Unlike the general verification, this approach uses a separate LLM call for
        each critical field, allowing more focused attention on the most important data.
        This is a more intensive verification approach that can be enabled or disabled
        via configuration.

        Args:
            report_text (str): Original report text from the PDF
            extraction (Dict): Extracted structured information to verify

        Returns:
            Dict: A dictionary mapping field names to detailed verification results
                 for each critical field, including:
                 - correct: Boolean indicating if the field is correct
                 - confidence: Confidence score for this field (0.0-1.0)
                 - actual_value: The verifier's assessment of the correct value
                 - evidence: Text from the report supporting the verification

        Note:
            This method is only executed if detailed_verification is enabled in the config.
            If disabled, it returns an empty dictionary.
        """
        if not self.config.detailed_verification:
            return {}

        # Critical fields that need special attention
        critical_fields = {
            "cancer_organ_type": "Identify the specific cancer organ type mentioned in the report",
            "cancer_subtype": "Identify the specific cancer subtype mentioned in the report",
            "figo_stage": "Identify any FIGO staging information in the report",
            "pathologic_stage": "Identify the pathologic staging information in the report",
        }

        field_results = {}

        # Verify each critical field individually
        for field, instruction in critical_fields.items():
            if field in extraction:
                try:
                    response = openai.chat.completions.create(
                        model=self.config.openai_model,  # Use simpler model for efficiency
                        messages=[
                            {"role": "system", "content": "You are a medical data verification specialist."},
                            {"role": "user", "content": f"""
                            From this pathology report excerpt:
                            ```
                            {report_text[:5000]}
                            ```

                            Task: {instruction}

                            The extraction system found: "{extraction[field]}"

                            Is this correct? Respond with a JSON object:
                            {{
                                "correct": true/false,
                                "confidence": (0.0-1.0),
                                "actual_value": "what you believe is the correct value",
                                "evidence": "text from the report supporting your conclusion"
                            }}
                            """}
                        ]
                    )

                    result = safe_json(response.choices[0].message.content)
                    field_results[field] = result

                except Exception as e:
                    logger.warning(f"Detailed verification for field '{field}' failed: {e}")
                    field_results[field] = {
                        "correct": True,  # Default to accepting the original
                        "confidence": 0.5,
                        "error": str(e)
                    }

        return field_results


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function for pathology report analyzer.

    This function serves as the entry point for the command-line interface of the
    Pathology Analyzer. It handles command-line argument parsing, configuration setup,
    and executes the main processing pipeline.

    Command-line arguments:
        --rebuild-cache: Force rebuilding of the embeddings cache
        --no-verify: Disable the verification step
        --detailed-verify: Enable detailed field-by-field verification
        --root-dir: Specify root directory for case folders
        --nccn-pdf: Specify path to NCCN guidelines PDF
        --api-key: Provide OpenAI API key
        --model: Specify OpenAI model to use

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pathology Report Analyzer")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild embeddings cache")
    parser.add_argument("--no-verify", action="store_true", help="Disable verification step")
    parser.add_argument("--detailed-verify", action="store_true", help="Enable detailed verification")
    parser.add_argument("--root-dir", type=str, help="Root directory containing case folders")
    parser.add_argument("--nccn-pdf", type=str, help="Path to NCCN guidelines PDF")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, help="OpenAI model to use")

    args = parser.parse_args()

    # Create configuration
    config = AnalyzerConfig()

    # Update config from command line arguments
    if args.root_dir:
        config.root_dir = Path(args.root_dir)
    if args.nccn_pdf:
        config.nccn_pdf_path = Path(args.nccn_pdf)
    if args.api_key:
        config.openai_api_key = args.api_key
        openai.api_key = args.api_key
    if args.model:
        config.openai_model = args.model
    if args.no_verify:
        config.verification_enabled = False
    if args.detailed_verify:
        config.detailed_verification = True

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Delete cache if rebuild requested
    if args.rebuild_cache and config.embeddings_cache_path.exists():
        logger.info(f"Deleting embeddings cache at {config.embeddings_cache_path}")
        config.embeddings_cache_path.unlink()

    # Start processing
    t0 = time.time()
    try:
        process_all_folders(config)
        logger.info(f"Finished in {time.time() - t0:.1f}s")
        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
