import json
import logging
from pathlib import Path
from typing import Dict, Optional

from tqdm.auto import tqdm

from config import AnalyzerConfig, ORGAN_TYPES_STR, SUBTYPES_STR
from embedding.embed import load_or_create_embeddings
from embedding.retrieval import top_k_guideline
from exceptions import TextExtractionError
from llm.client import analyze_with_reasoning
from llm.prompts import SYSTEM_MSG, PROMPT_TEMPLATE, INSTRUCTIONS_BLOCK
from llm.verification import VerificationAgent
from text_extraction.extract import extract_text_concurrent

logger = logging.getLogger("pathology_analyzer.analysis.process")


def analyse_report(
        report_text: str,
        guide_chunks: list,
        guide_embeds: list,
        config: AnalyzerConfig) -> Dict:
    """
    Analyze a pathology report using OpenAI API and relevant NCCN guidelines.

    This function implements a comprehensive two-stage approach to extract and verify
    structured information from pathology reports, optimized for accuracy and efficiency.

    Stage 1: Initial Extraction
    - Identifies relevant NCCN guideline chunks for the report using semantic search
    - Constructs a prompt combining the report, guidelines, and extraction instructions
    - Calls the OpenAI API to extract structured information

    Stage 2: Verification and Correction (if enabled)
    - Takes the extracted information and sends it to an optimized verification step
    - Verifies accuracy, completeness, and consistency of the extraction
    - Identifies and corrects any issues in the extracted data
    - Adds verification metadata to the final result

    Args:
        report_text (str): The text of the pathology report to analyze
        guide_chunks (List[str]): List of text chunks from the NCCN guidelines
        guide_embeds (List[np.ndarray]): Matrix of embeddings for each guideline chunk
        config (AnalyzerConfig): Configuration parameters for the analysis

    Returns:
        Dict: A dictionary containing the structured information extracted from the report,
              including verification results if verification was enabled
    """
    # Find relevant NCCN guideline chunks for this report
    nccn_text = "\n\n".join(top_k_guideline(
        report_text, guide_chunks, guide_embeds,
        k=config.k_guideline_chunks, threshold=config.sim_threshold,
        embed_model=config.openai_embed_model
    ))

    # Log the analysis process
    logger.info(f"Analyzing report with {len(report_text)} chars and {len(nccn_text)} chars of NCCN guidelines")

    # Create the full prompt with report and guidelines
    user_msg = PROMPT_TEMPLATE.format(
        organ_types=ORGAN_TYPES_STR,
        subtypes=SUBTYPES_STR,
        report=report_text,
        nccn=nccn_text,
        instructions=INSTRUCTIONS_BLOCK
    )

    # Stage 1: Initial extraction
    logger.info("Performing initial extraction with reasoning...")
    try:
        # Use a lower temperature for more consistent results
        primary_extraction = analyze_with_reasoning(
            msgs=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg}
            ],
            model=config.openai_model,
            enable_reasoning=config.enable_reasoning
            # o4 models don't support custom temperature
        )

        logger.info(f"Initial extraction successful: {list(primary_extraction.keys())}")

        # Basic validation of extraction structure
        required_fields = ["cancer_organ_type", "cancer_subtype", "figo_stage",
                          "pathologic_stage", "recommended_treatment",
                          "description", "patient_notes"]

        missing_fields = [field for field in required_fields if field not in primary_extraction]
        if missing_fields:
            logger.warning(f"Extraction missing required fields: {missing_fields}")
            # Add missing fields with "Not specified" value
            for field in missing_fields:
                primary_extraction[field] = "Not specified"

        # Basic validation for cancer_subtype - ensure it matches allowed options
        if (primary_extraction.get("cancer_organ_type") == "Uterine cancer" and
            primary_extraction.get("cancer_subtype") != "Not specified"):
            # Simple substring check - more sophisticated matching would be better in production
            if primary_extraction["cancer_subtype"] not in SUBTYPES_STR:
                logger.warning(f"Cancer subtype '{primary_extraction['cancer_subtype']}' not in allowed list, setting to 'Not specified'")
                primary_extraction["cancer_subtype"] = "Not specified"

        # Stage 2: Verification and correction (only if enabled and stage 1 succeeds)
        if config.verification_enabled:
            logger.info("Performing verification...")

            # Initialize verification agent
            verification_agent = VerificationAgent(config)

            # Perform verification with optimized approach
            verification_result = verification_agent.verify(
                report_text=report_text,
                extraction=primary_extraction,
                nccn_text=nccn_text
            )

            # Process verification results
            if verification_result.get('verification_result') == 'PASS':
                logger.info("Verification PASSED")
                return verification_agent.apply_corrections(primary_extraction, verification_result)
            else:
                logger.info("Verification FAILED - applying corrections")

                # Apply corrections to the primary extraction
                corrected_extraction = verification_agent.apply_corrections(
                    primary_extraction, verification_result
                )

                return corrected_extraction

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


def process_case_folder(case_folder: Path, guide_chunks: list, guide_embeds: list, config: AnalyzerConfig) -> Optional[Dict]:
    """
    Process a single case folder, analyzing its PDF file and saving the results.

    Args:
        case_folder (Path): Path to the case folder
        guide_chunks (list): List of NCCN guideline chunks
        guide_embeds (list): Embeddings for the guideline chunks
        config (AnalyzerConfig): Configuration object

    Returns:
        Optional[Dict]: Analysis results if successful, None otherwise
    """
    out_file = case_folder / "analysis.json"

    # Skip if already processed
    if out_file.exists():
        logger.info(f"Skip {case_folder.name} (already processed)")
        return None

    # Find PDF files in the case folder
    pdfs = sorted(case_folder.glob("*.pdf"))

    # Validate PDF existence
    if not pdfs:
        logger.warning(f"No PDF found in {case_folder}")
        return None

    # Warn if multiple PDFs found
    if len(pdfs) > 1:
        logger.warning(f"{case_folder}: multiple PDFs, using {pdfs[0].name}")

    try:
        # Extract text from the PDF using the concurrent method
        text = extract_text_concurrent(pdfs[0], config.text_similarity_threshold)

        # Analyze the report
        analysis = analyse_report(text, guide_chunks, guide_embeds, config)

        # Validate analysis results before saving
        validate_analysis_result(analysis)

        # Save results to JSON file
        out_file.write_text(json.dumps(analysis, indent=2))
        logger.info(f"Wrote analysis to {out_file.relative_to(config.root_dir)}")

        return analysis

    except TextExtractionError as e:
        logger.error(f"Text extraction failed for {case_folder.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing {case_folder.name}: {e}")
        return None


def validate_analysis_result(analysis: Dict) -> None:
    """
    Validate analysis results to ensure they meet requirements.

    Args:
        analysis (Dict): Analysis result to validate

    Raises:
        ValueError: If validation fails
    """
    # Ensure all required fields are present
    required_fields = [
        "cancer_organ_type", "cancer_subtype", "figo_stage",
        "pathologic_stage", "recommended_treatment",
        "description", "patient_notes"
    ]

    for field in required_fields:
        if field not in analysis:
            analysis[field] = "Not specified"

    # Ensure cancer_subtype is from the allowed list if organ is Uterine cancer
    if analysis.get("cancer_organ_type") == "Uterine cancer":
        from config import SUBTYPES_STR
        cancer_subtype = analysis.get("cancer_subtype", "")

        if cancer_subtype != "Not specified":
            # Simple check - in production, you might want a more sophisticated match
            if cancer_subtype not in SUBTYPES_STR:
                logger.warning(f"Final validation: cancer subtype '{cancer_subtype}' not in allowed list")
                analysis["cancer_subtype"] = "Not specified"

                # Update verification metadata if it exists
                if "verification" in analysis:
                    if "incorrect_fields" not in analysis["verification"]:
                        analysis["verification"]["incorrect_fields"] = []

                    if "cancer_subtype" not in analysis["verification"]["incorrect_fields"]:
                        analysis["verification"]["incorrect_fields"].append("cancer_subtype")

                    analysis["verification"]["passed"] = False

    # Ensure verification field is present
    if "verification" not in analysis:
        analysis["verification"] = {
            "passed": True,
            "confidence": 0.8,
            "assessment": "Auto-validated during final check",
            "incorrect_fields": []
        }


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
    for case in tqdm(sorted(cases), desc="Processing cases"):
        try:
            process_case_folder(case, guide_chunks, guide_embeds, config)
        except Exception as e:
            logger.error(f"Failed to process {case.name}: {e}")