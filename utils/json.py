import json
import re
import logging
from typing import Union, Dict, List

import openai

# Regular expression to extract JSON from text
json_block = re.compile(r'{.*}', re.S)

logger = logging.getLogger("pathology_analyzer.utils.json")


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
        if all(isinstance(item, dict) for item in text):  # Check if all items are dictionaries
            return text[0]  # Return the first dict in the list
        text = json.dumps(text)  # Otherwise convert list to JSON string

    # Ensure we're working with a string
    if not isinstance(text, str):  # Check if input is not a string
        text = str(text)  # Convert to string if needed

    # First attempt: direct JSON parsing
    try:
        return json.loads(text)  # Attempt to parse as JSON directly
    except json.JSONDecodeError:
        # Second attempt: find JSON block using regex
        match = json_block.search(text)  # Search for JSON-like pattern
        if match:
            try:
                return json.loads(match.group(0))  # Try to parse the matched text
            except json.JSONDecodeError:
                pass

    # Final attempt: use a model to repair the JSON
    try:
        repair_prompt = (
                "The following is meant to be a JSON object but is invalid.\n"
                "Fix syntax ONLY and return valid JSON with the same keys/values.\n\n"
                "-----\n" + text[:5000] + "\n-----"  # Limit to 5000 chars to avoid token limits
        )

        repair_resp = openai.chat.completions.create(  # Call OpenAI API for repair
            model="gpt-3.5-turbo",  # Use GPT-3.5-turbo model for repair
            messages=[{"role": "user", "content": repair_prompt}]  # Send the repair prompt
        )
        repaired = repair_resp.choices[0].message.content  # Extract repaired JSON from response
        return json.loads(repaired)  # Parse the repaired JSON
    except Exception as e:
        logger.error(f"JSON repair failed: {e}")  # Log the error
        return {
            "error": "Could not parse JSON response",
            "partial_content": text[:200] + "..." if len(text) > 200 else text
        }


def save_json(data: Union[Dict, List], file_path: str) -> None:
    """Save data to a JSON file with pretty formatting.

    Args:
        data (Union[Dict, List]): The data to save
        file_path (str): Path to the output JSON file

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Union[Dict, List]:
    """Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        Union[Dict, List]: The loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)