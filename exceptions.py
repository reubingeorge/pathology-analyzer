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


class TextComparisonError(PathologyAnalyzerError):
    """Raised when comparison between text extraction methods fails.

    This exception is raised when the system cannot reliably compare text extracted
    by different methods, or when the comparison process itself encounters an error.
    """
    pass