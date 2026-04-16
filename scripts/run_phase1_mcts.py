#!/usr/bin/env python3
"""Phase 1: MCTS-only GPU worker.

Runs MCTS exploration + trajectory collection + contrastive pair sampling.
Saves pairs to disk for Phase 2 (API-based analysis & synthesis).
No API calls — pure GPU computation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase1_mcts.py \
        --config configs/instruct_experiment.yaml \
        --shard_path data/shards/shard_0.jsonl \
        --output_dir data/phase1/gpu0 \
        --gpu_id 0
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crps.mcts.search import MCTSSearch
from crps.mcts.reward import MathRewardFunction
from crps.trajectory.collector import TrajectoryCollector
from crps.trajectory.sampler import ContrastivePairSampler
from crps.utils.llm import LLMInference
from crps.utils.math_verify import MathVerifier, AnswerExtractor


def setup_logging(output_dir: str, gpu_id: int):
    log_path = os.path.join(output_dir, "worker.log")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU{gpu_id}] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.gpu_id)
    logger = logging.getLogger(f"phase1.gpu{args.gpu_id}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    problems = []
    with open(args.shard_path) as f:
        for line in f:
            problems.append(json.loads(line))
    logger.info("Loaded %d problems", len(problems))

    # Only load MCTS model (local GPU)
    logger.info("Loading MCTS model: %s", config["mcts_model"]["name_or_path"])
    mcts_llm = LLMInference(
        model_name_or_path=config["mcts_model"]["name_or_path"],
        backend=config["mcts_model"]["backend"],
        temperature=config["mcts"]["temperature"],
        max_tokens=config["synthesis"]["max_tokens"],
        tensor_parallel_size=config["mcts_model"].get("tensor_parallel_size", 1),
    )
    _ = mcts_llm.generate("Hello", n=1)
    logger.info("MCTS model ready")

    verifier = MathVerifier()
    reward_fn = MathRewardFunction(verifier=verifier)
    collector = TrajectoryCollector()
    sampler = ContrastivePairSampler(k=config["trajectory"]["contrastive_pairs_per_problem"])

    # Output: pairs ready for analyst model
    pairs_path = os.path.join(args.output_dir, "pairs.jsonl")
    stats = {"total": len(problems), "processed": 0, "pairs_saved": 0,
             "correct_found": 0, "no_correct": 0, "errors": 0}

    # Resume
    processed = set()
    if os.path.exists(pairs_path):
        with open(pairs_path) as f:
            for line in f:
                d = json.loads(line)
                processed.add(d["problem"])
        logger.info("Resuming: %d problems already done", len(processed))
        stats["processed"] = len(processed)
        stats["pairs_saved"] = len(processed)  # approximate

    out = open(pairs_path, "a")
    t_start = time.time()

    # Filter out already-processed problems
    remaining = [item for item in problems if item["problem"] not in processed]
    logger.info("Remaining problems to process: %d", len(remaining))

    # Process in batches for better GPU utilization
    batch_size = config.get("mcts", {}).get("batch_size", 8)
    search = MCTSSearch(llm=mcts_llm, reward_fn=reward_fn, config=config["mcts"])

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_problems = [item["problem"] for item in batch]
        batch_gts = [item["ground_truth"] for item in batch]

        try:
            t0 = time.time()

            # Batch MCTS search — interleaved rollouts across problems
            trees = search.batch_search(batch_problems, batch_gts)

            for item, tree, gt in zip(batch, trees, batch_gts):
                problem = item["problem"]

                # Collect & stratify
                trajectories = collector.collect(
                    tree=tree, ground_truth=gt,
                    answer_extractor=AnswerExtractor, verifier=verifier,
                )
                stratified = collector.stratify(trajectories)
                n_correct = len(stratified.get("gold_positive", []))
                stats["correct_found"] += n_correct

                if n_correct == 0:
                    stats["no_correct"] += 1
                    stats["processed"] += 1
                    continue

                # Sample contrastive pairs
                anchor = collector.select_positive_anchor(stratified["gold_positive"])
                pairs = sampler.sample(problem, stratified, anchor)

                # Save each pair
                for pair in pairs:
                    record = {
                        "problem": problem,
                        "ground_truth": gt,
                        "source": item.get("source", "unknown"),
                        "contrast_type": pair.contrast_type,
                        "positive_steps": pair.positive.steps,
                        "negative_steps": pair.negative.steps,
                        "positive_answer": pair.positive.final_answer,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["pairs_saved"] += 1

                stats["processed"] += 1

            out.flush()
            dt = time.time() - t0

            if stats["processed"] % 5 == 0 or batch_start < batch_size:
                elapsed = time.time() - t_start
                rate = stats["processed"] / elapsed * 3600 if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] batch %.1fs (%.1fs/prob) | Pairs: %d | Rate: %.0f/hr | ETA: %.1fh",
                    stats["processed"], len(problems), dt, dt / len(batch),
                    stats["pairs_saved"], rate,
                    (len(problems) - stats["processed"]) / max(rate, 1) * 3600,
                )

        except Exception as e:
            stats["errors"] += len(batch)
            stats["processed"] += len(batch)
            logger.error("Batch error: %s\n%s", e, traceback.format_exc())

    out.close()
    dt_total = time.time() - t_start

    progress = os.path.join(args.output_dir, "progress.json")
    with open(progress, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("PHASE 1 GPU%d COMPLETE in %.1fh", args.gpu_id, dt_total / 3600)
    logger.info("  Processed: %d/%d, Pairs: %d, Errors: %d",
                stats["processed"], stats["total"], stats["pairs_saved"], stats["errors"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
