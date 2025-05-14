import datetime
import json
import logging
from typing import Dict

from tenacity import retry, stop_after_attempt, wait_exponential

from config import AnalyzerConfig
from llm.client import analyze_with_reasoning
from llm.prompts import (
    VERIFICATION_SYSTEM_MSG, COMBINED_VERIFICATION_PROMPT,
    INFERENCE_VALIDATION_COMBINED
)

logger = logging.getLogger("pathology_analyzer.llm.verification")


class VerificationAgent:
    """
    Enhanced verification agent for pathology report data with optimized API usage.

    This class provides detailed verification and cross-checking of extracted data
    from pathology reports. It uses a combined approach to verification to reduce
    API calls while ensuring high accuracy validation of extracted information.

    Key features:
    - Combined staging and field verification in a single API call
    - Strong focus on cancer subtype validation against allowed options
    - Efficient validation of medical consistency
    - Application of corrections to extraction results
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
        Verify the extraction results using a combined, efficient approach.

        This optimized method performs comprehensive verification with fewer API calls by:
        1. Combining staging verification with general verification in a single call
        2. Using a focused prompt that prioritizes the most critical aspects
        3. Providing clear instructions for the model to identify and correct issues

        Args:
            report_text (str): Original report text from the PDF
            extraction (Dict): Extracted structured information to verify
            nccn_text (str): Relevant NCCN guidelines text for reference

        Returns:
            Dict: A dictionary containing verification results with corrections if needed
        """
        try:
            # Use the combined verification approach
            verification_prompt = COMBINED_VERIFICATION_PROMPT.format(
                report=report_text[:7000],  # Use more text for better verification
                extracted_info=json.dumps(extraction, indent=2),
                nccn=nccn_text[:5000]
            )

            logger.info("Performing combined verification with focus on staging accuracy")

            # Use analyze_with_reasoning for verification
            verification_result = analyze_with_reasoning(
                msgs=[
                    {"role": "system", "content": VERIFICATION_SYSTEM_MSG},
                    {"role": "user", "content": verification_prompt}
                ],
                model=self.openai_model,
                enable_reasoning=self.config.enable_reasoning
                # o4 models don't support custom temperature
            )

            # Standardize result format
            standardized_result = {
                "verification_result": "PASS" if verification_result.get("passed", False) else "FAIL",
                "confidence_score": verification_result.get("confidence", 0.0),
                "field_verification": verification_result.get("field_issues", {}),
                "incorrect_fields": verification_result.get("incorrect_fields", []),
                "recommended_corrections": verification_result.get("recommended_corrections", {}),
                "overall_assessment": verification_result.get("assessment", "Verification completed"),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_used": self.openai_model
            }

            # Preserve detailed change justifications if provided
            if "change_justifications" in verification_result:
                standardized_result["change_justifications"] = verification_result["change_justifications"]

            # If no serious issues found, but we have the detailed verification enabled,
            # perform medical inference validation as an additional check
            if (self.config.detailed_verification and
                    standardized_result["verification_result"] == "PASS" and
                    standardized_result["confidence_score"] >= 0.8):

                logger.info("Primary verification passed, performing additional medical inference validation")
                inference_results = self.inference_validation(extraction)

                if inference_results.get("validation_result") == "FAIL":
                    logger.info("Medical inference validation found issues")

                    # Update verification result based on inference validation
                    standardized_result["verification_result"] = "FAIL"
                    standardized_result["confidence_score"] = min(
                        standardized_result["confidence_score"],
                        inference_results.get("confidence", 0.8)
                    )

                    # Add corrections from inference validation
                    if "corrections" in inference_results:
                        for field, value in inference_results["corrections"].items():
                            standardized_result["recommended_corrections"][field] = value
                            if field not in standardized_result["incorrect_fields"]:
                                standardized_result["incorrect_fields"].append(field)

                    # Update assessment
                    standardized_result["overall_assessment"] += " Medical inference validation identified issues: " + \
                                                                 inference_results.get("reasoning", "")

            return standardized_result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Return basic failure result
            return {
                "verification_result": "ERROR",
                "confidence_score": 0.0,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
                "overall_assessment": "Verification process failed due to technical error",
                "field_verification": {},
                "incorrect_fields": [],
                "recommended_corrections": {}
            }

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def inference_validation(self, extraction: Dict) -> Dict:
        """
        Perform inference-based validation using medical knowledge.

        This method validates the medical plausibility and consistency of the
        extracted data without relying on the report text. It uses the LLM's
        medical knowledge to identify potential issues based on relationships
        between fields.

        Args:
            extraction (Dict): Extracted structured information to validate

        Returns:
            Dict: Dictionary with validation results
        """
        logger.info("Performing medical inference validation")

        try:
            # Create prompt with extracted information
            validation_prompt = INFERENCE_VALIDATION_COMBINED.format(
                extracted_info=json.dumps(extraction, indent=2)
            )

            # Call the model with medical knowledge inference validation task
            validation_result = analyze_with_reasoning(
                msgs=[
                    {"role": "system", "content": "You are a medical expert validating pathology data consistency."},
                    {"role": "user", "content": validation_prompt}
                ],
                model=self.openai_model,
                enable_reasoning=True  # Always enable reasoning for medical inference
                # o4 models don't support custom temperature
            )

            # Add timestamp
            validation_result['timestamp'] = datetime.datetime.now().isoformat()
            validation_result['model_used'] = self.openai_model

            # Log any issues found
            if validation_result.get('validation_result') == 'FAIL':
                issues = validation_result.get('issues_found', [])
                if issues:
                    logger.info(f"Medical inference validation found {len(issues)} issues")
                    for issue in issues:
                        logger.info(f" - {issue}")

            return validation_result

        except Exception as e:
            logger.error(f"Medical inference validation failed: {e}")
            return {
                "validation_result": "ERROR",
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.datetime.now().isoformat()
            }

    def apply_corrections(self, extraction: Dict, verification: Dict) -> Dict:
        """
        Apply corrections from verification to the extracted data.

        This method takes the original extracted data and the verification results,
        and creates a corrected version of the extraction by applying any recommended
        corrections from the verification process. It also adds detailed verification
        metadata to the result, including very detailed explanations for any changes.

        Args:
            extraction (Dict): Original extraction data before corrections
            verification (Dict): Verification results containing recommended corrections
                               and verification metadata

        Returns:
            Dict: A new dictionary containing the corrected extraction data with all
                  original fields plus verification metadata
        """
        corrected = dict(extraction)

        # Create a change_explanations dict to store detailed explanations
        change_explanations = {}

        # Apply direct corrections if available
        corrections_applied = False
        if 'recommended_corrections' in verification:
            for field, value in verification['recommended_corrections'].items():
                if field in corrected:
                    original_value = corrected[field]
                    logger.info(f"Applying correction to field '{field}': {original_value} -> {value}")

                    # Store original value before changing
                    if original_value != value:
                        # Create detailed explanation for this change
                        explanation = f"Changed from '{original_value}' to '{value}'"

                        # Add field-specific details based on verification data
                        if 'field_verification' in verification and field in verification['field_verification']:
                            field_issues = verification['field_verification'][field].get('issues', '')
                            if field_issues:
                                explanation += f". Reason: {field_issues}"

                        # Add more detailed reasoning if available
                        if 'staging_verification' in verification and field in ['figo_stage', 'pathologic_stage']:
                            staging_details = verification.get('staging_verification', '')
                            if staging_details:
                                explanation += f". Staging details: {staging_details}"

                        # Add medical validation details if available
                        if 'medical_validation' in verification:
                            explanation += f". Medical validation: {verification['medical_validation']}"

                        # Store the explanation
                        change_explanations[field] = explanation

                    # Now apply the correction
                    corrected[field] = value
                    corrections_applied = True

        # Create verification metadata
        corrected['verification'] = {
            'passed': verification.get('verification_result') == 'PASS',
            'confidence': verification.get('confidence_score', 0.0),
            'assessment': verification.get('overall_assessment', 'Extraction verification completed'),
            'field_issues': verification.get('field_verification', {}),
            'incorrect_fields': verification.get('incorrect_fields', []),
            'timestamp': verification.get('timestamp', datetime.datetime.now().isoformat())
        }

        # Add the detailed change explanations if any changes were made
        if change_explanations:
            corrected['verification']['change_explanations'] = change_explanations

        # Special handling for cancer_subtype - ensure it's from the allowed list
        from config import SUBTYPES_STR
        if 'cancer_organ_type' in corrected and corrected['cancer_organ_type'] == 'Uterine cancer':
            current_subtype = corrected.get('cancer_subtype', '')

            # Check if current subtype is not in the allowed list
            # Simple check - more sophisticated matching would be better in production
            if current_subtype != "Not specified" and current_subtype not in SUBTYPES_STR:
                logger.warning(f"Cancer subtype '{current_subtype}' not in allowed list")

                # If we have a correction, it's already applied
                if 'cancer_subtype' not in verification.get('recommended_corrections', {}):
                    # Otherwise, set to "Not specified" if we can't match it
                    logger.info(f"Setting non-standard cancer subtype '{current_subtype}' to 'Not specified'")

                    # Store original value and create detailed explanation
                    if 'change_explanations' not in corrected['verification']:
                        corrected['verification']['change_explanations'] = {}

                    corrected['verification']['change_explanations']['cancer_subtype'] = (
                        f"Changed from '{current_subtype}' to 'Not specified'. Reason: The extracted cancer "
                        f"subtype '{current_subtype}' is not in the allowed list of subtypes for Uterine cancer. "
                        f"The system requires selecting only from predefined subtypes for consistency and "
                        f"accurate treatment recommendations. No close match was found in the allowed subtypes."
                    )

                    # Apply the correction
                    corrected['cancer_subtype'] = "Not specified"

                    # Update verification metadata
                    if 'field_issues' not in corrected['verification']:
                        corrected['verification']['field_issues'] = {}

                    corrected['verification']['field_issues']['cancer_subtype'] = {
                        "correct": False,
                        "confidence": 0.0,
                        "issues": f"Non-standard subtype '{current_subtype}' not in allowed list"
                    }

                    if 'incorrect_fields' not in corrected['verification']:
                        corrected['verification']['incorrect_fields'] = []

                    if 'cancer_subtype' not in corrected['verification']['incorrect_fields']:
                        corrected['verification']['incorrect_fields'].append('cancer_subtype')

        return corrected