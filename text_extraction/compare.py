import logging
from typing import Tuple, Optional
from difflib import SequenceMatcher
import statistics

from utils.text import normalize_for_comparison
from exceptions import TextComparisonError

logger = logging.getLogger("pathology_analyzer.text_extraction.compare")


def compare_extracted_texts(text1: str, text2: str, similarity_threshold: float = 0.8) -> Tuple[float, bool]:
    """
    Compare two extracted texts and determine if they are sufficiently similar.

    Args:
        text1 (str): First extracted text
        text2 (str): Second extracted text
        similarity_threshold (float): Threshold above which texts are considered similar

    Returns:
        Tuple[float, bool]: (similarity score, whether similarity exceeds threshold)
    """
    # Check if either text is empty
    if not text1 or not text2:
        return 0.0, False

    # Normalize texts for comparison
    norm_text1 = normalize_for_comparison(text1)
    norm_text2 = normalize_for_comparison(text2)

    # Calculate similarity ratio
    similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()

    return similarity, similarity >= similarity_threshold


def analyze_text_differences(text1: str, text2: str) -> dict:
    """
    Analyze differences between two extracted texts.

    Args:
        text1 (str): First extracted text
        text2 (str): Second extracted text

    Returns:
        dict: Analysis of differences including:
            - character_count_diff: Difference in character count
            - word_count_diff: Difference in word count
            - similarity_score: Overall similarity score
            - line_by_line_similarity: Statistics on line-by-line similarity
            - problem_areas: Sections with low similarity
    """
    result = {
        "character_count": {
            "text1": len(text1),
            "text2": len(text2),
            "difference": len(text1) - len(text2),
            "ratio": len(text1) / max(1, len(text2))
        },
        "word_count": {
            "text1": len(text1.split()),
            "text2": len(text2.split()),
            "difference": len(text1.split()) - len(text2.split()),
            "ratio": len(text1.split()) / max(1, len(text2.split()))
        },
        "similarity_score": SequenceMatcher(None,
                                            normalize_for_comparison(text1),
                                            normalize_for_comparison(text2)).ratio(),
        "line_by_line_similarity": {},
        "problem_areas": []
    }

    # Split into lines and compare line by line
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')

    line_similarities = []

    # Compare each line in text1 to its best match in text2
    for i, line1 in enumerate(lines1):
        if not line1.strip():
            continue

        best_similarity = 0
        best_match_idx = -1

        for j, line2 in enumerate(lines2):
            if not line2.strip():
                continue

            similarity = SequenceMatcher(None,
                                         normalize_for_comparison(line1),
                                         normalize_for_comparison(line2)).ratio()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = j

        if best_match_idx >= 0:
            line_similarities.append(best_similarity)

            # If similarity is low, record as a problem area
            if best_similarity < 0.7:
                result["problem_areas"].append({
                    "line_number_text1": i + 1,
                    "text1_line": line1[:100] + ("..." if len(line1) > 100 else ""),
                    "best_match_line_number_text2": best_match_idx + 1,
                    "text2_line": lines2[best_match_idx][:100] + ("..." if len(lines2[best_match_idx]) > 100 else ""),
                    "similarity": best_similarity
                })

    # Calculate statistics on line similarities
    if line_similarities:
        result["line_by_line_similarity"] = {
            "min": min(line_similarities),
            "max": max(line_similarities),
            "mean": statistics.mean(line_similarities),
            "median": statistics.median(line_similarities)
        }

        # Only include standard deviation if we have enough data points
        if len(line_similarities) > 1:
            result["line_by_line_similarity"]["std_dev"] = statistics.stdev(line_similarities)

    return result


def select_best_text(text1: str, text2: str, text3: Optional[str] = None) -> Tuple[str, str]:
    """
    Select the best text from multiple extraction methods.

    This function compares texts from different extraction methods and selects the one
    that is most likely to be correct. If three texts are provided, it uses a majority
    vote approach.

    Args:
        text1 (str): Text from first extraction method (e.g., PyMuPDF)
        text2 (str): Text from second extraction method (e.g., PyPDF2)
        text3 (Optional[str]): Text from third extraction method (e.g., OCR)

    Returns:
        Tuple[str, str]: (selected text, method name that produced it)
    """
    methods = ["method1", "method2", "method3"]
    texts = [text1, text2]
    if text3 is not None:
        texts.append(text3)

    # If any text is empty, remove it
    valid_texts = [(text, method) for text, method in zip(texts, methods[:len(texts)]) if text and len(text) > 100]

    if not valid_texts:
        raise TextComparisonError("No valid texts to compare")

    if len(valid_texts) == 1:
        # Only one valid text
        return valid_texts[0]

    # Compare each pair of texts
    similarities = []
    for i in range(len(valid_texts)):
        for j in range(i + 1, len(valid_texts)):
            similarity, _ = compare_extracted_texts(valid_texts[i][0], valid_texts[j][0])
            similarities.append((i, j, similarity))

    # If we have 3 valid texts, use majority vote
    if len(valid_texts) == 3 and len(similarities) == 3:
        # Find the two most similar texts
        similarities.sort(key=lambda x: x[2], reverse=True)
        if similarities[0][2] > 0.7:  # If similarity is high enough
            most_similar_pair = [similarities[0][0], similarities[0][1]]
            # Return the longer of the two most similar texts
            if len(valid_texts[most_similar_pair[0]][0]) >= len(valid_texts[most_similar_pair[1]][0]):
                return valid_texts[most_similar_pair[0]]
            else:
                return valid_texts[most_similar_pair[1]]

    # Otherwise, return the longest text
    valid_texts.sort(key=lambda x: len(x[0]), reverse=True)
    return valid_texts[0]