import logging
import numpy as np
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential
from embedding.embed import embed

logger = logging.getLogger("pathology_analyzer.embedding.retrieval")


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def top_k_guideline(query: str, guide_chunks: List[str], guide_embeds: np.ndarray,
                    k: int = 8, threshold: float = 0.25, embed_model: str = "text-embedding-3-large") -> List[str]:
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
        embed_model (str): Model to use for creating embeddings

    Returns:
        List[str]: List of the most relevant guideline text chunks, ordered by relevance
    """
    # Embed the query
    q_emb = embed(query, embed_model)

    # Calculate similarity to all guideline chunks
    sims = np.array([cosine(e, q_emb) for e in guide_embeds])

    # Get indices of top k chunks above threshold
    idx = [i for i in sims.argsort()[::-1] if sims[i] >= threshold][:k]

    selected_chunks = [guide_chunks[i] for i in idx]
    logger.info(
        f"Selected {len(selected_chunks)} guideline chunks with similarity scores from {sims[idx[-1]]:.3f} to {sims[idx[0]]:.3f}")

    # Return the corresponding chunks
    return selected_chunks