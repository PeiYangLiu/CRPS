#!/usr/bin/env python3
"""Fast evaluation using vLLM with few-shot support.

Usage:
    # Base model (few-shot)
    python scripts/run_eval_vllm.py --model_path /path/to/base --model_name base --few_shot --benchmarks gsm8k math
    
    # Fine-tuned model (zero-shot)
    python scripts/run_eval_vllm.py --model_path /path/to/crps --model_name crps --benchmarks gsm8k math
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_vllm")

# ── Few-shot exemplars ───────────────────────────────────────────────

GSM8K_EXEMPLARS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{6}."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is \\boxed{39}."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is \\boxed{8}."),
]

MATH_EXEMPLARS = [
    ("Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
     "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}$."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating the two expressions, we can find $n$: $30n=480 \\implies n=\\frac{480}{30}=\\boxed{16}$."),
    ("If the system of equations \\begin{align*} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $6y-9x=-\\frac{3a}{2}$. Since we also know that $6y-9x=b$, we have $-\\frac{3a}{2}=b \\implies \\frac{a}{b}=\\boxed{-\\frac{2}{3}}$."),
]


def load_gsm8k_test():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for item in ds:
        match = re.search(r"####\s*(.+)", item["answer"])
        gt = match.group(1).strip().replace(",", "") if match else item["answer"].strip()
        problems.append({"problem": item["question"], "ground_truth": gt})
    return problems


def load_math_test():
    from datasets import load_dataset
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    problems = []
    for item in ds:
        # Use non-greedy nested brace matching for \boxed{}
        match = re.search(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", item["solution"])
        gt = match.group(1).strip() if match else item["solution"].strip()
        problems.append({"problem": item["problem"], "ground_truth": gt})
    return problems


def extract_answer(text, is_few_shot=False):
    # For few-shot (base model): truncate at continuation
    if is_few_shot:
        for marker in ["\nProblem:", "\n\nProblem:"]:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx]

    # \boxed{...} with nested brace support — take LAST match
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", text)
    if matches:
        return matches[-1].strip()
    # "The answer is ..."
    m = re.search(r"(?:The answer is|the answer is)\s*[:\s]*(.*?)(?:\.|$)", text)
    if m:
        return m.group(1).strip()
    # Last number
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1]
    return ""


def normalize(ans):
    ans = ans.strip()
    # LaTeX normalization
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\left{", "{").replace("\\right}", "}")
    ans = ans.replace("\\left|", "|").replace("\\right|", "|")
    # Remove spaces around operators and punctuation
    ans = re.sub(r"\s+", "", ans)
    # Remove dollar signs, percent
    ans = ans.replace("$", "").replace("%", "")
    # Try numeric conversion
    if "/" in ans and "\\" not in ans:
        try:
            parts = ans.split("/")
            if len(parts) == 2:
                return f"{float(parts[0]) / float(parts[1]):.6f}"
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return f"{float(ans):.6f}"
    except ValueError:
        return ans.lower()


def check_answer(predicted, ground_truth):
    if not predicted:
        return False
    # Normalized string match
    if normalize(predicted) == normalize(ground_truth):
        return True
    # Try sympy symbolic equivalence
    try:
        from sympy import simplify, sympify
        if simplify(sympify(predicted) - sympify(ground_truth)) == 0:
            return True
    except Exception:
        pass
    return False


def build_prompts(problems, benchmark, few_shot):
    """Build all prompts for batch generation."""
    prefix = ""
    if few_shot:
        exemplars = GSM8K_EXEMPLARS if "gsm" in benchmark.lower() else MATH_EXEMPLARS
        for q, a in exemplars:
            prefix += f"Problem: {q}\nSolution: {a}\n\n"

    prompts = []
    for item in problems:
        if few_shot:
            prompts.append(f"{prefix}Problem: {item['problem']}\nSolution:")
        else:
            # DeepSeek official format
            prompts.append(f"{item['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.")
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math"])
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--output_dir", default="eval_results_vllm")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_name = args.model_name or Path(args.model_path).name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading model with vLLM: %s (tp=%d)", args.model_path, args.tp)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp,
              trust_remote_code=True, max_model_len=4096)
    params = SamplingParams(temperature=0.0, max_tokens=4096)
    logger.info("Model loaded: %s", model_name)

    all_results = {"model": model_name, "few_shot": args.few_shot}

    for bench in args.benchmarks:
        if bench == "gsm8k":
            problems = load_gsm8k_test()
        elif bench == "math":
            problems = load_math_test()
        else:
            continue

        if args.max_samples:
            problems = problems[:args.max_samples]

        if args.shard_id is not None and args.num_shards is not None:
            import math
            chunk = math.ceil(len(problems) / args.num_shards)
            start = args.shard_id * chunk
            end = min(start + chunk, len(problems))
            problems = problems[start:end]
            logger.info("Shard %d/%d: problems [%d:%d] (%d total)",
                        args.shard_id, args.num_shards, start, end, len(problems))

        prompts = build_prompts(problems, bench, args.few_shot)
        logger.info("Evaluating %s: %d problems (few_shot=%s)", bench.upper(), len(problems), args.few_shot)

        t0 = time.time()
        # vLLM batch generation - much faster than sequential
        outputs = llm.generate(prompts, params)
        dt = time.time() - t0

        correct = 0
        details = []
        for item, output in zip(problems, outputs):
            response = output.outputs[0].text
            predicted = extract_answer(response, is_few_shot=args.few_shot)
            is_correct = check_answer(predicted, item["ground_truth"])
            if is_correct:
                correct += 1
            details.append({
                "problem": item["problem"],
                "ground_truth": item["ground_truth"],
                "predicted": predicted,
                "correct": is_correct,
                "response": response,
            })

        accuracy = correct / len(problems) * 100
        logger.info("%s RESULT: %.1f%% (%d/%d) in %.0fs (%.1f q/s)",
                    bench.upper(), accuracy, correct, len(problems), dt, len(problems) / dt)

        all_results[bench] = {"accuracy": accuracy, "correct": correct, "total": len(problems)}

        # Save details
        fs_tag = "_fewshot" if args.few_shot else ""
        shard_tag = f"_s{args.shard_id}" if args.shard_id is not None else ""
        with open(os.path.join(args.output_dir, f"{model_name}_{bench}{fs_tag}{shard_tag}_details.jsonl"), "w") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Summary
    summary_path = os.path.join(args.output_dir, f"{model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("")
    logger.info("=" * 50)
    logger.info("RESULTS: %s (few_shot=%s)", model_name, args.few_shot)
    logger.info("=" * 50)
    for bench in args.benchmarks:
        if bench in all_results:
            r = all_results[bench]
            logger.info("  %-10s: %.1f%% (%d/%d)", bench.upper(), r["accuracy"], r["correct"], r["total"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
