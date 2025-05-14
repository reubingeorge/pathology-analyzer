"""
main.py - Entry point for the Pathology Analyzer application
"""

import argparse
import logging
import os
import time
from pathlib import Path

import openai

from config import AnalyzerConfig
from utils.logger import setup_logging
from embedding.embed import clear_embedding_cache
from analysis.process import process_all_folders
from analysis.utils import collect_results_parallel


def main():
    """Main function for pathology report analyzer.

    This function serves as the entry point for the command-line interface of the
    Pathology Analyzer. It handles command-line argument parsing, configuration setup,
    and executes the main processing pipeline.

    Command-line arguments:
        --rebuild-cache: Force rebuilding of the embeddings cache
        --no-verify: Disable the verification step
        --detailed-verify: Enable detailed field-by-field verification
        --no-reasoning: Disable reasoning mode for OpenAI models
        --root-dir: Specify root directory for case folders
        --nccn-pdf: Specify path to NCCN guidelines PDF
        --api-key: Provide OpenAI API key
        --model: Specify OpenAI model to use
        --text-similarity: Similarity threshold for text extraction comparison
        --summary: Only generate summary of existing results without processing
        --parallel: Number of parallel workers for summary generation

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pathology Report Analyzer")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild embeddings cache")
    parser.add_argument("--no-verify", action="store_true", help="Disable verification step")
    parser.add_argument("--detailed-verify", action="store_true", help="Enable detailed verification")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning mode for OpenAI models")
    parser.add_argument("--root-dir", type=str, help="Root directory containing case folders")
    parser.add_argument("--nccn-pdf", type=str, help="Path to NCCN guidelines PDF")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, help="OpenAI model to use")
    parser.add_argument("--text-similarity", type=float, help="Text similarity threshold (0.0-1.0)")
    parser.add_argument("--summary", action="store_true", help="Only generate summary of existing results")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers for summary")

    args = parser.parse_args()

    # Initialize logging
    logger = setup_logging()

    # Create configuration
    config = AnalyzerConfig()

    # Update config from command line arguments
    if args.root_dir:
        config.root_dir = Path(args.root_dir)
    if args.nccn_pdf:
        config.nccn_pdf_path = Path(args.nccn_pdf)
    if args.api_key:
        config.openai_api_key = args.api_key
        openai.api_key = args.api_key
    if args.model:
        config.openai_model = args.model
    if args.no_verify:
        config.verification_enabled = False
    if args.detailed_verify:
        config.detailed_verification = True
    if args.no_reasoning:
        config.enable_reasoning = False
    if args.text_similarity is not None:
        config.text_similarity_threshold = args.text_similarity

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # If only generating summary
    if args.summary:
        logger.info(f"Generating summary from existing results in {config.root_dir}")
        summary = collect_results_parallel(config.root_dir, max_workers=args.parallel)
        summary_path = config.root_dir / "analysis_summary.json"

        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")
        logger.info(f"Processed {summary['processed_cases']} of {summary['total_cases']} cases")
        logger.info(f"Successful: {summary['successful_cases']}, Verification passed: {summary['verification_passed']}")
        return 0

    # Delete cache if rebuild requested
    if args.rebuild_cache and config.embeddings_cache_path.exists():
        logger.info(f"Deleting embeddings cache at {config.embeddings_cache_path}")
        config.embeddings_cache_path.unlink()
        clear_embedding_cache()

    # Start processing
    t0 = time.time()
    try:
        process_all_folders(config)
        elapsed = time.time() - t0
        logger.info(f"Finished in {elapsed:.1f}s")

        # Generate summary after processing
        logger.info("Generating summary of results")
        summary = collect_results_parallel(config.root_dir, max_workers=args.parallel)
        summary_path = config.root_dir / "analysis_summary.json"

        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")
        logger.info(f"Processed {summary['processed_cases']} of {summary['total_cases']} cases")
        logger.info(f"Successful: {summary['successful_cases']}, Verification passed: {summary['verification_passed']}")

        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())