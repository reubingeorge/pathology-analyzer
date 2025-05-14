import logging
import pickle
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import AnalyzerConfig
from exceptions import EmbeddingError
from text_extraction.extract import extract_text_concurrent
from utils.text import split_by_tokens

logger = logging.getLogger("pathology_analyzer.embedding")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
@lru_cache(maxsize=1024)  # Cache embeddings to avoid re-computing
def embed(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """Create an embedding vector for text using OpenAI's embedding model.

    This function converts a text string into a vector representation (embedding)
    using OpenAI's text-embedding-3-large model. The resulting embedding can be used
    for semantic similarity comparisons and information retrieval.

    The function is decorated with:
    - @retry: Automatically retries failed API calls up to 3 times with exponential backoff
    - @lru_cache: Caches results to avoid redundant API calls for the same text

    Args:
        text (str): The text to create an embedding for
        model (str): The embedding model to use

    Returns:
        np.ndarray: A numpy array containing the embedding vector

    Raises:
        EmbeddingError: If the embedding creation fails after retries
    """
    try:
        return np.array(  # Convert response to numpy array
            openai.embeddings.create(  # Call OpenAI embeddings API
                model=model,  # Specify embedding model
                input=text  # Provide text to embed
            ).data[0].embedding  # Extract the embedding vector
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise EmbeddingError(f"Failed to create embedding: {e}")


def clear_embedding_cache():
    """Clear the embedding function's LRU cache.

    This can be useful when you want to force re-computation of embeddings,
    or when you change the embedding model.
    """
    embed.cache_clear()
    logger.info("Embedding cache cleared")


def load_or_create_embeddings(config: AnalyzerConfig) -> Tuple[List[str], np.ndarray]:
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
                cache_data = pickle.load(f)  # Load cached data
                guide_chunks = cache_data['chunks']  # Extract chunks from cache
                guide_embeds = cache_data['embeds']  # Extract embeddings from cache
                logger.info(f"Loaded {len(guide_chunks)} chunks from cache")
                return guide_chunks, guide_embeds
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}. Regenerating...")

    # If cache doesn't exist or loading failed, generate embeddings
    logger.info("Loading NCCN guideline PDF...")
    try:
        # Extract text using the concurrent extraction method
        guideline_raw = extract_text_concurrent(config.nccn_pdf_path, config.text_similarity_threshold)

        # Split guidelines into manageable chunks
        guide_chunks = split_by_tokens(guideline_raw, config.chunk_token_size)  # Split into chunks
        logger.info(f"Chunked NCCN guideline into {len(guide_chunks)} pieces ≤ {config.chunk_token_size} tokens each")

        # Embed all chunks for later similarity search
        logger.info("Embedding guideline chunks... this runs once and will be cached")
        from tqdm.auto import tqdm
        guide_embeds = np.vstack([  # Stack embeddings into a matrix
            embed(chunk, config.openai_embed_model) for chunk in tqdm(guide_chunks, desc="Embed NCCN")
        ])

        # Save embeddings to cache
        logger.info(f"Saving embeddings to cache: {config.embeddings_cache_path}")
        with open(config.embeddings_cache_path, 'wb') as f:
            pickle.dump({'chunks': guide_chunks, 'embeds': guide_embeds}, f)

        return guide_chunks, guide_embeds
    except Exception as e:
        logger.error(f"Failed to process NCCN guidelines: {e}")
        raise