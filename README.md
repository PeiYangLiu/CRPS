# CRPS: Contrastive Reasoning Path Synthesis

**A framework for synthesizing high-quality mathematical reasoning data through contrastive analysis of MCTS-guided search trajectories.**

CRPS implements a **decoupled explorer-analyst paradigm**: a specialized reasoning model (explorer) generates diverse reasoning trajectories via MCTS, while a capable analyst model performs contrastive analysis between correct and incorrect paths to synthesize refined, critique-informed solutions. The resulting datasets enable supervised fine-tuning (SFT) of target base models with significant improvements in mathematical reasoning—achieving comparable performance to baselines trained on 20× more data.

📦 **Pre-built Dataset**: [PeiyangLiu/CRPS-30K on HuggingFace](https://huggingface.co/datasets/PeiyangLiu/CRPS-30K) — 30K synthesized reasoning paths ready for SFT, no pipeline execution needed.

---

## Pipeline Overview

The pipeline operates in two phases for maximum efficiency:

```
Phase 1 (GPU-only):  Seed Problems → MCTS Exploration → Contrastive Pair Sampling
Phase 2 (API-only):  Pairs → Analyst Contrastive Analysis → Path Synthesis → Verification
Training:            Verified Paths → SFT on Target Base Model
```

**Key Components:**

1. **MCTS Exploration** (Phase 1) — Explorer model (Qwen2.5-Math-7B-Instruct) runs UCT-based tree search on seed problems
2. **Trajectory Stratification** (Phase 1) — Collect gold positives, hard negatives (P ∝ visit count), and soft negatives
3. **Contrastive Analysis** (Phase 2) — Analyst model (gpt-5-mini) performs dual-granularity analysis (global strategic + local step-wise)
4. **Path Synthesis** (Phase 2) — Analyst synthesizes new solutions incorporating success patterns while avoiding failure modes
5. **Verification** (Phase 2) — Filter synthesized paths via answer matching with SymPy equivalence checking
6. **SFT Training** — Fine-tune target base model (DeepSeekMath-7B-Base, LLaMA3-8B, or Mistral-7B) on verified data

---

## Installation

```bash
pip install -r requirements.txt
```

For multi-GPU training with DeepSpeed:
```bash
pip install deepspeed>=0.14.0
```

---

## Quick Start

### Phase 1: MCTS Exploration (GPU)

Run MCTS on seed problems and collect contrastive pairs:

```bash
python scripts/run_phase1_mcts.py \
    --config configs/instruct_experiment.yaml \
    --output_dir ./data/phase1 \
    --gpu_id 0
```

For multi-GPU parallel execution:
```bash
for gpu in 0 1 2 3 4 5 6 7; do
    python scripts/run_phase1_mcts.py \
        --config configs/instruct_experiment.yaml \
        --output_dir ./data/phase1 \
        --gpu_id $gpu &
done
```

### Phase 2: Analysis + Synthesis (API)

Run contrastive analysis and path synthesis via the analyst model:

```bash
python scripts/run_phase2_api.py \
    --config configs/instruct_experiment.yaml \
    --pairs_dir ./data/phase1 \
    --output_dir ./data/phase2 \
    --max_workers 16
```

### Training

Fine-tune a target base model on the verified synthesized data:

```bash
deepspeed --num_gpus 8 scripts/run_training.py \
    --config configs/default.yaml \
    --data_path ./data/phase2/verified.jsonl \
    --output_dir ./outputs/crps-deepseek-7b
```

### Evaluation

Evaluate using vLLM batch inference:

```bash
python scripts/run_eval_vllm.py \
    --model_path ./outputs/crps-deepseek-7b/final \
    --benchmark math \
    --output_dir ./eval_results
```

---

## Configuration

All hyperparameters are centralized in YAML config files. Key sections:

| Section | Description |
|---|---|
| `mcts_model` | Explorer model path and backend (vLLM) |
| `analysis_model` | Analyst model for contrastive analysis |
| `synthesis_model` | Analyst model for path synthesis |
| `mcts` | UCT parameters: `c_puct=1.4`, `max_depth=16`, `max_actions_per_node=3`, `num_rollouts=10` |
| `trajectory` | Number of contrastive pairs K=10 per problem |
| `training` | SFT: AdamW (β₁=0.9, β₂=0.95), lr=2e-5, cosine schedule, warmup 3%, batch=128, 3 epochs |

See `configs/instruct_experiment.yaml` for the full experiment configuration.

---

## Project Structure

```
crps/
├── configs/
│   ├── default.yaml              # Default hyperparameters
│   ├── instruct_experiment.yaml   # Full experiment config
│   ├── full_experiment.yaml       # Full-scale experiment config
│   └── ds_zero3.json             # DeepSpeed ZeRO-3 config
├── crps/                          # Core library
│   ├── mcts/                      # MCTS search (UCT, node, reward)
│   ├── trajectory/                # Trajectory collection & contrastive pair sampling
│   ├── analysis/                  # Dual-granularity contrastive analysis
│   ├── synthesis/                 # Pattern-informed path synthesis & verification
│   ├── training/                  # SFT dataset & trainer
│   ├── prompts/                   # All prompt templates
│   └── utils/                     # LLM inference, math verification (SymPy)
├── scripts/
│   ├── run_phase1_mcts.py         # Phase 1: MCTS exploration (GPU)
│   ├── run_phase2_api.py          # Phase 2: Analysis + synthesis (API)
│   ├── run_training.py            # SFT training with DeepSpeed
│   └── run_eval_vllm.py           # vLLM batch evaluation
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@misc{liu2026learningcontrastssynthesizingreasoning,
      title={Learning from Contrasts: Synthesizing Reasoning Paths from Diverse Search Trajectories}, 
      author={Peiyang Liu and Zhirui Chen and Xi Wang and Di Liang and Youru Li and Zhi Cai and Wei Ye},
      year={2026},
      eprint={2604.11365},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.11365}, 
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
