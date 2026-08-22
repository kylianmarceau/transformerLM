# CSC3043S Assignment 1: Training a Transformer Language Model

## Installation and versions

The recorded local preprocessing environment used:

- Python 3.14.6
- PyTorch 2.13.0 locally; the AfriLink container version was not recorded
- NumPy 2.5.1
- "regex" 2026.7.19
- Matplotlib 3.11.1

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 numpy==2.5.1 regex==2026.7.19 matplotlib==3.11.1

AfriLink runs also require the afrilink  package and an API key.

## Reproducing the experiments

Run commands from the repository root. Place the supplied datasets at:

  
datasets/TinyStoriesV2-GPT4-train.txt
datasets/TinyStoriesV2-GPT4-valid.txt

### Tokenizers and encoded corpora

python scripts/vocab_study.py --input datasets/TinyStoriesV2-GPT4-valid.txt --vocab-sizes 1000 2000 4000 8000 16000 --reserved-test-documents 2000 --output-dir results/vocab_study
python scripts/encode_corpus.py --train datasets/TinyStoriesV2-GPT4-train.txt --validation datasets/TinyStoriesV2-GPT4-valid.txt --vocab-sizes 4000 1000
python generate_q4.py
  
### AfriLink training runs
  
export AFRILINK_API_KEY="your-key"
  
Learning-rate sweep:
  
python submit_run.py --name lr_1e_4 --group lr_sweep --lr 1e-4 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_3e_4 --group lr_sweep --lr 3e-4 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_1e_3 --group lr_sweep --lr 1e-3 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_3e_3 --group lr_sweep --lr 3e-3 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_1e_2 --group lr_sweep --lr 1e-2 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_3e_2 --group lr_sweep --lr 3e-2 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_1e_1 --group lr_sweep --lr 1e-1 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_3e_1 --group lr_sweep --lr 3e-1 --steps 1000 --checkpoint-interval 1000
python submit_run.py --name lr_1e0 --group lr_sweep --lr 1e0 --steps 1000 --checkpoint-interval 1000

Baseline, ablations, vocabulary comparison, and final model:
  
python submit_run.py --name baseline_5000 --group baseline --lr 3e-3 --steps 5000 --checkpoint-interval 5000
python submit_run.py --name no_rmsnorm_5000 --group ablations --lr 3e-3 --steps 5000 --no-rmsnorm --checkpoint-interval 5000
python submit_run.py --name no_rmsnorm_lr_1e_3_5000 --group ablations --lr 1e-3 --steps 5000 --no-rmsnorm --checkpoint-interval 5000
python submit_run.py --name nope_5000 --group ablations --lr 3e-3 --steps 5000 --no-rope --checkpoint-interval 5000
python submit_run.py --name vocab_1000_5000 --group vocabulary --lr 3e-3 --steps 5000 --vocab-size 1000 --checkpoint-interval 5000
python submit_run.py --name final_vocab4000_20000 --group final --lr 3e-3 --steps 20000 --vocab-size 4000 --checkpoint-interval 5000
  


### Evaluation and generation

Final validation evaluation:
  
python src/evaluate.py --checkpoint afrilink_results/final/final_vocab4000_20000/checkpoints/final_vocab4000_20000_step_020000.pt --data-path datasets/encoded/valid_vocab_4000.npy --split validation --batch-size 32 --output afrilink_results/final/final_vocab4000_20000/validation_metrics.json
  
Vocabulary comparison:
  
python src/evaluate.py --checkpoint afrilink_results/baseline/baseline_5000/checkpoints/baseline_5000_step_005000.pt --data-path datasets/encoded/valid_vocab_4000.npy --split validation --batch-size 32 --output results/q15_vocab4000_validation_metrics.json
python src/evaluate.py --checkpoint afrilink_results/vocabulary/vocab_1000_5000/checkpoints/vocab_1000_5000_step_005000.pt --data-path datasets/encoded/valid_vocab_1000.npy --split validation --batch-size 32 --output results/q15_vocab1000_validation_metrics.json

KV-cache correctness and generation:
  
python -m unittest tests.test_kv_cache -v
python generate_q7.py
python generate_q19.py
  
Do not rerun the held-out test set. The one-time Q18 result is saved in "afrilink_results/final/final_vocab4000_20000/test_metrics.json".

python scripts/benchmark_bpe.py
python scripts/benchmark_generation.py --device cpu
python scripts/evaluate_position_loss.py
python scripts/make_report_plots.py
python scripts/build_report_metrics.py

## Report figure sources

| Figure | Source data | Producing script |
|---|---|---|
| Figure 1: vocabulary compression | "results/vocab_study/vocab_study.csv" | "scripts/vocab_study.py" |
| Figure 2: KV-cache throughput | "results/q7_generation_throughput.json" | "scripts/benchmark_generation.py" |
| Figure 3: learning-rate sweep | "afrilink_results/lr_sweep/*/*.jsonl" | "scripts/make_report_plots.py" |
| Figure 4: RMSNorm ablation | Baseline and no-RMSNorm JSONL logs | "scripts/make_report_plots.py" |
| Figure 5: NoPE versus RoPE | Baseline and "nope_5000" JSONL logs | "scripts/make_report_plots.py" |
| Figure 6: position-wise loss | "results/q17_position_losses.json" | "scripts/evaluate_position_loss.py" and "scripts/make_report_plots.py" |
