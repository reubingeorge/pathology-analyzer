import logging
import time
from typing import List, Dict

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from exceptions import OpenAIAPIError
from llm.prompts import SYSTEM_MSG_WITH_REASONING
from utils.json import safe_json

logger = logging.getLogger("pathology_analyzer.llm.client")


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((IOError, ValueError))
)
def call_openai(msgs: List[Dict], model: str = "o4-mini", enable_reasoning: bool = True,
                temperature: float = None) -> Dict:
    """Send a request to OpenAI API with reasoning and parse the response as JSON.

    This function sends a conversation to the OpenAI Chat Completions API
    with reasoning encouraged through prompt engineering, and processes the response
    into a structured JSON format. It handles errors and retry logic for
    robustness against transient API issues.

    The function is configured to retry up to 4 times with exponential backoff
    when encountering IOError or ValueError exceptions.

    Args:
        msgs (List[Dict]): List of message dictionaries in the OpenAI Chat format,
                          each containing at minimum 'role' and 'content' keys
        model (str): The OpenAI model to use (default: "o4-mini")
        enable_reasoning (bool): Whether to enhance prompts with reasoning instructions
        temperature (float, optional): Temperature parameter for the API call. If None, uses model default.
                                      Note: o4 models don't support custom temperature values.

    Returns:
        Dict: The parsed JSON response from the OpenAI API

    Raises:
        OpenAIAPIError: If the API call fails after retries or returns unparseable content
    """
    start_time = time.time()
    logger.info(f"Calling OpenAI API with model {model} (reasoning: {enable_reasoning})")

    try:
        # Prepare API call parameters - o4 models don't support custom temperature
        params = {
            "model": model,
            "messages": msgs,
            "response_format": {"type": "json_object"}  # Ensure JSON response
        }

        # Only add temperature if explicitly provided, not None, and not using an o4 model
        if temperature is not None and not model.startswith("o4"):
            params["temperature"] = temperature
            logger.info(f"Using custom temperature: {temperature}")
        else:
            logger.info("Using default temperature (model doesn't support custom temperature)")

        # Make the API call with prepared parameters
        response = openai.chat.completions.create(**params)

        elapsed_time = time.time() - start_time
        logger.info(f"OpenAI API call completed in {elapsed_time:.2f} seconds")

        # Extract and parse response content
        content = response.choices[0].message.content

        # Ensure we get valid JSON even if there's text before or after
        result = safe_json(content)

        # Validate required fields if this is a pathology analysis result
        required_fields = ["cancer_organ_type", "cancer_subtype", "figo_stage",
                           "pathologic_stage", "recommended_treatment"]

        missing_fields = [field for field in required_fields if field in result and not result[field]]

        # If we have missing fields, replace empty values with "Not specified"
        for field in missing_fields:
            if field in result and not result[field]:
                logger.warning(f"Empty value in field {field}, replacing with 'Not specified'")
                result[field] = "Not specified"

        return result

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise OpenAIAPIError(f"Error calling OpenAI API: {e}")


def enhance_system_message_for_reasoning(system_msg: str) -> str:
    """Enhance a system message to encourage step-by-step reasoning.

    This function adds prompts for step-by-step thinking to the system message
    to improve the reasoning capabilities of the model.

    Args:
        system_msg (str): The original system message

    Returns:
        str: Enhanced system message that encourages reasoning
    """
    reasoning_prefix = (
        "This is a reasoning task that requires careful analysis. "
        "Think step-by-step before providing your answer. "
        "Break down the problem, consider each part separately, "
        "and ensure your reasoning is thorough and accurate.\n\n"
    )

    return reasoning_prefix + system_msg


def analyze_with_reasoning(
        msgs: List[Dict],
        model: str = "o4-mini",
        enable_reasoning: bool = True,
        temperature: float = None
) -> Dict:
    """Analyze a pathology report with explicit reasoning through prompt engineering.

    This function enhances the standard OpenAI API call with additional
    reasoning capabilities by adjusting prompts to encourage step-by-step thinking.

    Args:
        msgs (List[Dict]): List of message dictionaries in the OpenAI Chat format
        model (str): The OpenAI model to use
        enable_reasoning (bool): Whether to enhance prompts with reasoning instructions
        temperature (float, optional): Temperature parameter for the API call. If None, uses model default.
                                      Note: o4 models don't support custom temperature values.

    Returns:
        Dict: The parsed JSON response
    """
    # Create a deep copy of messages to avoid modifying the original
    import copy
    enhanced_msgs = copy.deepcopy(msgs)

    # Replace standard system message with reasoning-enhanced version if present
    if enable_reasoning and enhanced_msgs and enhanced_msgs[0]["role"] == "system":
        if model.startswith("o4"):
            # For o4 models with reasoning enabled, use the special system message
            enhanced_msgs[0]["content"] = SYSTEM_MSG_WITH_REASONING
        else:
            # For other models, enhance the existing system message
            enhanced_msgs[0]["content"] = enhance_system_message_for_reasoning(enhanced_msgs[0]["content"])

    # For all models, add explicit reasoning instructions to the user message
    if enable_reasoning and len(enhanced_msgs) > 1:
        for i, msg in enumerate(enhanced_msgs):
            if msg["role"] == "user":
                # Add reasoning instruction but keep critical format requirements at the end
                current_content = msg["content"]

                # Only add reasoning prompt if it's not already there
                if "think step-by-step" not in current_content.lower():
                    # Find the position after instructions but before any format specifications
                    insert_pos = current_content.find("JSON format")

                    if insert_pos == -1:
                        # If no format spec found, add to the end
                        enhanced_msgs[i]["content"] = current_content + "\n\nPlease think step-by-step before answering."
                    else:
                        # Insert before format specifications
                        enhanced_msgs[i]["content"] = (
                            current_content[:insert_pos] +
                            "\nPlease think step-by-step before answering. Break down your analysis into clear steps.\n\n" +
                            current_content[insert_pos:]
                        )
                break

    # Call the API with the modified messages, but don't set temperature for o4 models
    if model.startswith("o4"):
        # o4 models don't support custom temperature
        return call_openai(enhanced_msgs, model, False, None)
    else:
        # Other models can use custom temperature
        return call_openai(enhanced_msgs, model, False, temperature)