#!/usr/bin/env python3
"""Phase 2: API-based contrastive analysis + synthesis.

Reads contrastive pairs from Phase 1, calls analyst model API for
analysis and synthesis, then verifies results. No GPU needed.

Supports high concurrency with ThreadPoolExecutor.

Usage:
    python scripts/run_phase2_api.py \
        --config configs/instruct_experiment.yaml \
        --pairs_dir data/phase1 \
        --output_dir data/phase2 \
        --max_workers 16
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crps.analysis.contrastive import ContrastiveAnalyzer
from crps.synthesis.synthesizer import PathSynthesizer
from crps.synthesis.verifier import SynthesisVerifier
from crps.trajectory.collector import Trajectory
from crps.trajectory.sampler import ContrastivePair
from crps.utils.llm import LLMInference
from crps.utils.math_verify import MathVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase2")


def load_all_pairs(pairs_dir: str) -> list[dict]:
    """Load all pairs from Phase 1 GPU outputs."""
    all_pairs = []
    for fname in sorted(os.listdir(pairs_dir)):
        fpath = os.path.join(pairs_dir, fname)
        if os.path.isdir(fpath):
            pairs_file = os.path.join(fpath, "pairs.jsonl")
            if os.path.exists(pairs_file):
                with open(pairs_file) as f:
                    for line in f:
                        all_pairs.append(json.loads(line))
                logger.info("  %s: %d pairs", fname, sum(1 for _ in open(pairs_file)))
    logger.info("Total pairs loaded: %d", len(all_pairs))
    return all_pairs


def record_to_pair(record: dict) -> ContrastivePair:
    """Convert a JSON record back into a ContrastivePair object."""
    positive = Trajectory(
        steps=record["positive_steps"],
        is_correct=True,
        reward=1.0,
        visit_count=1,
        length=len(record["positive_steps"]),
        max_q_value=1.0,
        final_answer=record.get("positive_answer"),
    )
    negative = Trajectory(
        steps=record["negative_steps"],
        is_correct=record["contrast_type"] != "hard",
        reward=0.0 if record["contrast_type"] == "hard" else 1.0,
        visit_count=1,
        length=len(record["negative_steps"]),
        max_q_value=0.0,
        final_answer=None,
    )
    return ContrastivePair(
        problem=record["problem"],
        positive=positive,
        negative=negative,
        contrast_type=record["contrast_type"],
    )


def process_single_pair(
    record: dict,
    analyzer: ContrastiveAnalyzer,
    synthesizer: PathSynthesizer,
    syn_verifier: SynthesisVerifier,
) -> list[dict]:
    """Process one pair: analyze → synthesize → verify. Returns verified results."""
    pair = record_to_pair(record)
    gt = record["ground_truth"]
    results = []

    try:
        # Contrastive analysis
        critique = analyzer.analyze(pair)
        if critique is None:
            return []

        # Synthesis
        path = synthesizer.synthesize(critique)
        if path is None:
            return []

        # Verify
        verified = syn_verifier.verify(path, gt)
        if verified.is_verified:
            results.append({
                "problem": record["problem"],
                "solution": "\n".join(verified.steps),
                "answer": verified.final_answer,
                "source": record.get("source", "unknown"),
                "contrast_type": record["contrast_type"],
            })
    except Exception as e:
        logger.debug("Error processing pair: %s", e)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: API analysis + synthesis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs_dir", required=True, help="Phase 1 output directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Concurrent API calls")
    parser.add_argument("--target_size", type=int, default=None,
                        help="Stop after getting this many verified paths")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load all pairs from Phase 1
    logger.info("Loading pairs from %s", args.pairs_dir)
    all_records = load_all_pairs(args.pairs_dir)

    if not all_records:
        logger.error("No pairs found! Run Phase 1 first.")
        return

    # Initialize API models (no GPU needed)
    logger.info("Initializing analysis model: %s", config["analysis_model"]["name_or_path"])
    analysis_llm = LLMInference(
        model_name_or_path=config["analysis_model"]["name_or_path"],
        backend=config["analysis_model"]["backend"],
        temperature=config["analysis"]["temperature"],
        max_tokens=config["analysis"]["max_tokens"],
        api_base_url=config["analysis_model"].get("api_base_url"),
        api_key=config["analysis_model"].get("api_key"),
    )

    logger.info("Initializing synthesis model: %s", config["synthesis_model"]["name_or_path"])
    synthesis_llm = LLMInference(
        model_name_or_path=config["synthesis_model"]["name_or_path"],
        backend=config["synthesis_model"]["backend"],
        temperature=config["synthesis"]["temperature"],
        max_tokens=config["synthesis"]["max_tokens"],
        api_base_url=config["synthesis_model"].get("api_base_url"),
        api_key=config["synthesis_model"].get("api_key"),
    )

    analyzer = ContrastiveAnalyzer(llm=analysis_llm)
    synthesizer = PathSynthesizer(llm=synthesis_llm,
                                   temperature=config["synthesis"]["temperature"])
    syn_verifier = SynthesisVerifier(math_verifier=MathVerifier())

    target = args.target_size or config["dataset"].get("target_size", len(all_records))

    # Process with concurrent API calls
    verified_path = os.path.join(args.output_dir, "verified_paths.jsonl")

    # Resume support: skip already-processed pairs
    already_done = set()
    total_verified = 0
    if os.path.exists(verified_path):
        with open(verified_path) as f:
            for line in f:
                d = json.loads(line)
                already_done.add(d["problem"][:80] + d.get("contrast_type", ""))
                total_verified += 1
        logger.info("Resuming: %d verified paths already done", total_verified)

    out = open(verified_path, "a")
    total_processed = 0
    t_start = time.time()

    logger.info("Processing %d pairs with %d concurrent workers (target: %d verified)",
                len(all_records), args.max_workers, target)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i, record in enumerate(all_records):
            if total_verified >= target:
                break
            record_key = record["problem"][:80] + record.get("contrast_type", "")
            if record_key in already_done:
                continue
            future = executor.submit(
                process_single_pair, record, analyzer, synthesizer, syn_verifier
            )
            futures[future] = i

        for future in as_completed(futures):
            total_processed += 1
            results = future.result()
            for r in results:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                total_verified += 1

            if total_processed % 20 == 0:
                elapsed = time.time() - t_start
                rate = total_processed / elapsed * 3600
                logger.info(
                    "[%d/%d processed] Verified: %d/%d target | Rate: %.0f/hr | Elapsed: %.1fm",
                    total_processed, len(all_records), total_verified, target,
                    rate, elapsed / 60,
                )

            if total_verified >= target:
                logger.info("Target reached! Cancelling remaining tasks...")
                for f in futures:
                    f.cancel()
                break

    out.close()
    dt = time.time() - t_start

    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info("  Time: %.1f minutes", dt / 60)
    logger.info("  Processed: %d pairs", total_processed)
    logger.info("  Verified: %d paths", total_verified)
    logger.info("  Output: %s", verified_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
