import re
from typing import List
import tiktoken

# Initialize tokenizer once for reuse
_enc = tiktoken.get_encoding("cl100k_base")  # This is NOT future-proof. But currently correct


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
    tokens = _enc.encode(text)  # Encode the text into tokens
    chunks = [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)]  # Split into chunks
    return [_enc.decode(chunk) for chunk in chunks]  # Decode each chunk back to text


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing line breaks.

    Args:
        text (str): The text to clean

    Returns:
        str: The cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    # Remove whitespace at the beginning and end of lines
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)

    return text.strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison by removing all whitespace, punctuation, and converting to lowercase.

    This is useful for comparing text extracted by different methods, where formatting
    and whitespace might differ but the actual content is the same.

    Args:
        text (str): The text to normalize

    Returns:
        str: The normalized text
    """
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove all whitespace
    text = re.sub(r'\s+', '', text)

    # Convert to lowercase
    return text.lower()


def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings using normalized Levenshtein distance.

    This function computes a similarity score between 0 and 1, where 1 means identical
    and 0 means completely different. It normalizes the texts before comparison to focus
    on content rather than formatting.

    Args:
        text1 (str): First text string
        text2 (str): Second text string

    Returns:
        float: Similarity score between 0 and 1
    """
    from difflib import SequenceMatcher

    # Normalize texts for comparison
    norm_text1 = normalize_for_comparison(text1)
    norm_text2 = normalize_for_comparison(text2)

    # Calculate similarity ratio
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()