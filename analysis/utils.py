import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("pathology_analyzer.analysis.utils")


def extract_statistics(analysis_results: Dict) -> Dict:
    """
    Extract detailed statistical information from pathology analysis results for a single case.

    This function processes the analysis results dictionary to extract key metrics and presence indicators
    for important clinical fields. It identifies which fields are present in the analysis, tracks important
    staging information (FIGO and pathologic staging), treatment recommendations, and verification status.

    Args:
        analysis_results (Dict): Analysis results from a single pathology case containing extracted
                               information like cancer type, staging, and treatment recommendations

    Returns:
        Dict: A dictionary containing statistics about the analysis with the following keys:
            - fields_present (List[str]): List of fields successfully extracted from the report
            - verification_passed (bool): Whether the analysis passed verification checks
            - verification_confidence (float): Confidence score of the verification (0.0-1.0)
            - has_figo_stage (bool): Whether FIGO staging information is present
            - has_pathologic_stage (bool): Whether pathologic staging information is present
            - has_treatment_recommendation (bool): Whether treatment recommendations are present
            - has_error (bool): Whether an error occurred during analysis

    Note:
        Fields are considered present only if they exist in the analysis_results and have a value
        other than "Not specified".
    """
    stats = {
        "fields_present": [],
        "verification_passed": False,
        "verification_confidence": 0.0,
        "has_figo_stage": False,
        "has_pathologic_stage": False,
        "has_treatment_recommendation": False,
    }

    # Check which fields are present
    for field in ["cancer_organ_type", "cancer_subtype", "figo_stage",
                  "pathologic_stage", "recommended_treatment"]:
        if field in analysis_results and analysis_results[field] != "Not specified":
            stats["fields_present"].append(field)

            # Set specific flags for important fields
            if field == "figo_stage":
                stats["has_figo_stage"] = True
            elif field == "pathologic_stage":
                stats["has_pathologic_stage"] = True
            elif field == "recommended_treatment":
                stats["has_treatment_recommendation"] = True

    # Check verification status
    if "verification" in analysis_results:
        stats["verification_passed"] = analysis_results["verification"].get("passed", False)
        stats["verification_confidence"] = analysis_results["verification"].get("confidence", 0.0)

    # Check error status
    stats["has_error"] = "error" in analysis_results

    return stats


def collect_results(root_dir: Path) -> Dict:
    """
    Collect and comprehensively summarize results from all processed pathology cases.

    This function walks through all subdirectories in the specified root directory, looking for
    case folders containing analysis.json files. It aggregates statistics across all cases,
    tracking success rates, verification metrics, cancer types, staging information, and errors.

    Args:
        root_dir (Path): Root directory containing case folders, where each folder represents
                       a single pathology case with an analysis.json file containing results

    Returns:
        Dict: A detailed summary of analysis results across all cases with the following keys:
            - total_cases (int): Total number of case folders found
            - processed_cases (int): Number of cases with analysis.json files
            - successful_cases (int): Number of cases processed without errors
            - verification_passed (int): Number of cases that passed verification
            - cancer_types (Dict[str, int]): Distribution of cancer types across cases
            - has_figo_stage (int): Number of cases with FIGO staging information
            - has_pathologic_stage (int): Number of cases with pathologic staging
            - has_treatment_recommendation (int): Number of cases with treatment recommendations
            - average_verification_confidence (float): Average verification confidence across successful cases
            - detailed_verification_stats (Dict): Additional verification statistics
            - errors (List[Dict]): List of errors encountered during processing

    Note:
        This function processes cases sequentially. For parallel processing of large datasets,
        use collect_results_parallel() instead.
    """
    summary = {
        "total_cases": 0,
        "processed_cases": 0,
        "successful_cases": 0,
        "verification_passed": 0,
        "cancer_types": {},
        "has_figo_stage": 0,
        "has_pathologic_stage": 0,
        "has_treatment_recommendation": 0,
        "average_verification_confidence": 0.0,
        "detailed_verification_stats": {},
        "errors": [],
    }

    # Find all case folders
    case_folders = [p for p in root_dir.iterdir()
                    if p.is_dir() and not p.name.startswith('.')]
    summary["total_cases"] = len(case_folders)

    # Process each folder
    for folder in case_folders:
        analysis_path = folder / "analysis.json"

        if not analysis_path.exists():
            continue

        summary["processed_cases"] += 1

        try:
            # Load analysis results
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)

            # Extract statistics
            stats = extract_statistics(analysis)

            # Update summary
            if not stats["has_error"]:
                summary["successful_cases"] += 1

                # Cancer types
                cancer_type = analysis.get("cancer_organ_type", "Not specified")
                if cancer_type != "Not specified":
                    summary["cancer_types"][cancer_type] = summary["cancer_types"].get(cancer_type, 0) + 1

                # Verification statistics
                if stats["verification_passed"]:
                    summary["verification_passed"] += 1
                summary["average_verification_confidence"] += stats["verification_confidence"]

                # Field presence
                if stats["has_figo_stage"]:
                    summary["has_figo_stage"] += 1
                if stats["has_pathologic_stage"]:
                    summary["has_pathologic_stage"] += 1
                if stats["has_treatment_recommendation"]:
                    summary["has_treatment_recommendation"] += 1
            else:
                summary["errors"].append({
                    "case": folder.name,
                    "error": analysis.get("error", "Unknown error")
                })

        except Exception as e:
            logger.error(f"Error processing results from {folder.name}: {e}")
            summary["errors"].append({
                "case": folder.name,
                "error": str(e)
            })

    # Calculate averages
    if summary["successful_cases"] > 0:
        summary["average_verification_confidence"] /= summary["successful_cases"]

    return summary


def collect_results_parallel(root_dir: Path, max_workers: int = 4) -> Dict:
    """
    Collect and summarize results from all processed pathology cases using parallel processing.

    This function provides the same functionality as collect_results() but uses a ThreadPoolExecutor
    to process multiple case folders concurrently, significantly improving performance for large datasets.
    It walks through all subdirectories in the specified root directory, processing each case folder in
    parallel up to the specified maximum number of worker threads.

    Args:
        root_dir (Path): Root directory containing case folders, where each folder represents
                       a single pathology case with an analysis.json file containing results
        max_workers (int, optional): Maximum number of concurrent worker threads to use for processing.
                                   Defaults to 4, which balances performance with resource usage.

    Returns:
        Dict: A detailed summary of analysis results across all cases with the following keys:
            - total_cases (int): Total number of case folders found
            - processed_cases (int): Number of cases with analysis.json files
            - successful_cases (int): Number of cases processed without errors
            - verification_passed (int): Number of cases that passed verification
            - cancer_types (Dict[str, int]): Distribution of cancer types across cases
            - has_figo_stage (int): Number of cases with FIGO staging information
            - has_pathologic_stage (int): Number of cases with pathologic staging
            - has_treatment_recommendation (int): Number of cases with treatment recommendations
            - average_verification_confidence (float): Average verification confidence across successful cases
            - detailed_verification_stats (Dict): Additional verification statistics
            - errors (List[Dict]): List of errors encountered during processing

    Note:
        The ThreadPoolExecutor handles task distribution and result aggregation automatically.
        This function is recommended for datasets with more than 10 cases or when processing
        needs to be completed quickly. The max_workers parameter should be tuned based on
        available CPU cores and system resources.
    """
    # Find all case folders
    case_folders = [p for p in root_dir.iterdir()
                    if p.is_dir() and not p.name.startswith('.')]

    # Function to process a single case folder
    def process_folder(folder: Path) -> Optional[Dict]:
        analysis_path = folder / "analysis.json"

        if not analysis_path.exists():
            return None

        try:
            # Load analysis results
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)

            # Extract statistics and return with folder name
            stats = extract_statistics(analysis)
            return {
                "folder": folder.name,
                "analysis": analysis,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error processing results from {folder.name}: {e}")
            return {
                "folder": folder.name,
                "error": str(e)
            }

    # Process folders in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(process_folder, folder): folder for folder in case_folders}
        for future in concurrent.futures.as_completed(future_to_folder):
            result = future.result()
            if result:
                results.append(result)

    # Compile summary
    summary = {
        "total_cases": len(case_folders),
        "processed_cases": len(results),
        "successful_cases": 0,
        "verification_passed": 0,
        "cancer_types": {},
        "has_figo_stage": 0,
        "has_pathologic_stage": 0,
        "has_treatment_recommendation": 0,
        "average_verification_confidence": 0.0,
        "detailed_verification_stats": {},
        "errors": [],
    }

    # Process results
    for result in results:
        if "error" in result:
            summary["errors"].append({
                "case": result["folder"],
                "error": result["error"]
            })
            continue

        stats = result["stats"]
        analysis = result["analysis"]

        if not stats["has_error"]:
            summary["successful_cases"] += 1

            # Cancer types
            cancer_type = analysis.get("cancer_organ_type", "Not specified")
            if cancer_type != "Not specified":
                summary["cancer_types"][cancer_type] = summary["cancer_types"].get(cancer_type, 0) + 1

            # Verification statistics
            if stats["verification_passed"]:
                summary["verification_passed"] += 1
            summary["average_verification_confidence"] += stats["verification_confidence"]

            # Field presence
            if stats["has_figo_stage"]:
                summary["has_figo_stage"] += 1
            if stats["has_pathologic_stage"]:
                summary["has_pathologic_stage"] += 1
            if stats["has_treatment_recommendation"]:
                summary["has_treatment_recommendation"] += 1
        else:
            summary["errors"].append({
                "case": result["folder"],
                "error": analysis.get("error", "Unknown error")
            })

    # Calculate averages
    if summary["successful_cases"] > 0:
        summary["average_verification_confidence"] /= summary["successful_cases"]

    return summary