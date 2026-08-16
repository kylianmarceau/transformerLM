from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch 
import torch.nn.functional as F
from contextlib import nullcontext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import BatchPrefetcher, MemoryMappedTokenData
from src.model import TransformerConfig, TransformerLM

def build_optimizer(model,learning_rate: float,weight_decay: float = 0.1,beta1: float = 0.9,beta2: float = 0.95,epsilon: float = 1e-8,):
    """build the adam optimzer for 5.1"""

    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    if weight_decay < 0:
        raise ValueError("weight decay cannot be negative")

    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("AdamW beta values must be in [0, 1)")

    if epsilon <= 0:
        raise ValueError("AdamW epsilon must be positive")

    decay_parameters=[]
    no_decay_parameters=[]

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [{"params": decay_parameters, "weight_decay": weight_decay,},{"params": no_decay_parameters, "weight_decay": 0.0,},]

    return torch.optim.AdamW(parameter_groups,lr=learning_rate,betas=(beta1, beta2),eps=epsilon,)

def build_scheduler(optimizer,warmup_steps: int,total_steps: int,minimum_lr_ratio: float = 0.1,):
    # Build the linear warmup and cosine decay scheduler for 5.2
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")

    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")

    if not 0 < minimum_lr_ratio <= 1:
        raise ValueError("minimum_lr_ratio must be between 0 and 1")

    decay_steps = total_steps - warmup_steps

    def learning_rate_multiplier(step: int):
        # the first optimiser step starts above zero and the last warmup step reaches the configured maximum learning rate
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        # cosine period is tied to this runs actual number of steps
        if decay_steps == 1:
            return minimum_lr_ratio

        progress = (step - warmup_steps)/(decay_steps-1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=learning_rate_multiplier,)

def clip_gradients(model, max_norm: float = 1.0):
    """clip gradients and return their total norm before clipping"""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

def mixed_precision_context(device, enabled: bool = True):
    # use bf16 autocast on cuda and fp32 on anythign else 
    device = torch.device(device)

    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,)

    return nullcontext()
# ----

@dataclass
class TrainingConfig:
    train_path: str = str(PROJECT_ROOT / "datasets" / "encoded" / "train_vocab_4000.npy")
    validation_path: str = str(PROJECT_ROOT / "datasets" / "encoded" / "valid_vocab_4000.npy")
    output_dir: str = str(PROJECT_ROOT / "runs")
    run_name: str = "baseline"

    vocab_size: int = 4000
    context_length: int = 256
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1344
    rope_theta: float = 10_000.0
    use_qk_norm: bool = True
    use_rmsnorm: bool = True
    use_rope: bool = True
    ffn_type: str = "swiglu"

    batch_size: int = 32
    prefetch_batches: int = 2
    total_steps: int = 5_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    warmup_steps: int = 200
    minimum_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0

    evaluation_interval: int = 100
    evaluation_batches: int = 20
    validation_reserved_documents: int = 2_000
    document_separator_token_id: int = 256
    checkpoint_interval: int = 500
    seed: int = 3043
    device: str = "auto"
    use_bf16: bool = True
    resume: str | None = None

def validate_training_config(config: TrainingConfig):
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    if config.context_length <= 0:
        raise ValueError("context_length must be positive")

    if config.n_layers <= 0:
        raise ValueError("n_layers must be positive")

    if config.d_model <= 0:
        raise ValueError("d_model must be positive")

    if config.n_heads <= 0 or config.d_model % config.n_heads != 0:
        raise ValueError("n_heads must be positive and divide d_model")

    if config.d_ff <= 0:
        raise ValueError("d_ff must be positive")

    if config.rope_theta <= 0:
        raise ValueError("rope_theta must be positive")

    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.prefetch_batches <= 0:
        raise ValueError("prefetch_batches must be positive")

    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")

    if not 0 <= config.adam_beta1 < 1 or not 0 <= config.adam_beta2 < 1:
        raise ValueError("AdamW beta values must be in [0, 1)")

    if config.adam_epsilon <= 0:
        raise ValueError("adam_epsilon must be positive")

    if not 0 <= config.warmup_steps < config.total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps - 1")

    if not 0 < config.minimum_lr_ratio <= 1:
        raise ValueError("minimum_lr_ratio must be in (0, 1]")

    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    if config.evaluation_interval <= 0:
        raise ValueError("evaluation_interval must be positive")

    if config.evaluation_batches <= 0:
        raise ValueError("evaluation_batches must be positive")

    if config.validation_reserved_documents < 0:
        raise ValueError("validation_reserved_documents cannot be negative")

    if (config.validation_reserved_documents and not 0 <= config.document_separator_token_id < config.vocab_size):
        raise ValueError("document_separator_token_id must be in the vocabulary")

    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    if config.ffn_type not in {"swiglu", "relu"}:
        raise ValueError("ffn_type must be 'swiglu' or 'relu'")


def choose_device(requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    return device

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_model(config: TrainingConfig):
    model_config = TransformerConfig(vocab_size=config.vocab_size,context_length=config.context_length,n_layers=config.n_layers,d_model=config.d_model,n_heads=config.n_heads,d_ff=config.d_ff,rope_theta=config.rope_theta,use_qk_norm=config.use_qk_norm,use_rmsnorm=config.use_rmsnorm,use_rope=config.use_rope,ffn_type=config.ffn_type,)
    return TransformerLM(model_config)

@torch.no_grad()
def evaluate_model(model,validation_batches,device,use_bf16: bool,):
    was_training = model.training
    model.eval()
    losses = []

    for inputs, targets in validation_batches:
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        with mixed_precision_context(device, enabled=use_bf16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),targets.reshape(-1),)

        losses.append(loss.float())

    model.train(was_training)

    return torch.stack(losses).mean().item()

def append_jsonl(path: Path, record: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")

def write_json(path: Path, payload: dict[str, object]):
    """Atomically write a JSON object so interrupted writes cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",encoding="utf-8",)
    os.replace(temporary_path, path)

def truncate_jsonl_after_step(path: Path, completed_step: int):
    """Discard log rows newer than a resumed checkpoint to avoid duplicates."""
    if not path.exists():
        return

    retained_lines = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
            record_step = int(record["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            # An interruption can leave only the final append partly written
            # D
            # dropping that tail is safe because checkpoints are written later
            if line_number == len(lines):
                break
            raise ValueError(
                f"Invalid log record at {path}:{line_number}"
            ) from error

        if record_step <= completed_step:
            retained_lines.append(json.dumps(record))

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    contents = "\n".join(retained_lines)
    if contents:
        contents += "\n"
    temporary_path.write_text(contents, encoding="utf-8")
    os.replace(temporary_path, path)

def validate_resume_config(saved: dict[str, object], current: TrainingConfig):
    """make sure a checkpoint resumes the same experiment configuration"""
    allowed_to_change = {"train_path","validation_path","output_dir","run_name","resume","device",}
    current_values = asdict(current)
    mismatches = []

    for name, current_value in current_values.items():
        if name in allowed_to_change:
            continue

        if name not in saved or saved[name] != current_value:
            mismatches.append(f"{name}: checkpoint={saved.get(name)!r}, current={current_value!r}")

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Resume configuration does not match checkpoint: {details}")

def checkpoint_payload(model,optimizer,scheduler,completed_step: int,tokens_processed: int,elapsed_seconds: float,train_generator: torch.Generator,config: TrainingConfig,train_generator_state: torch.Tensor | None = None,):
    rng_state = {"python": random.getstate(),"numpy": np.random.get_state(),"torch": torch.get_rng_state(),"train_generator": (train_generator.get_state() if train_generator_state is None else train_generator_state),"cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),}

    return {"model": model.state_dict(),"optimizer": optimizer.state_dict(),"scheduler": scheduler.state_dict(),"step": completed_step,"tokens_processed": tokens_processed,"elapsed_seconds": elapsed_seconds,"rng_state": rng_state,"config": asdict(config),}

def save_checkpoint(path: Path, payload: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

def load_checkpoint(path: str | Path,model,optimizer,scheduler,train_generator: torch.Generator,):
    checkpoint = torch.load(path,map_location="cpu",weights_only=False,)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    train_generator.set_state(rng_state["train_generator"].cpu())

    if torch.cuda.is_available() and rng_state["cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])

    return checkpoint

def run_training(config: TrainingConfig):
    validate_training_config(config)
    seed_everything(config.seed)
    device = choose_device(config.device)

    output_dir = Path(config.output_dir)
    log_path = output_dir / f"{config.run_name}.jsonl"
    step_log_path = output_dir / f"{config.run_name}_steps.jsonl"
    config_path = output_dir / f"{config.run_name}_config.json"
    checkpoint_dir = output_dir / "checkpoints"

    if config.resume is None and (log_path.exists() or step_log_path.exists()):
        raise FileExistsError(
            f"Logs already exist for run {config.run_name!r}. Choose a new run_name."
        )

    train_data = MemoryMappedTokenData(config.train_path)
    validation_data = MemoryMappedTokenData(config.validation_path,reserved_trailing_documents=config.validation_reserved_documents,document_separator_token_id=config.document_separator_token_id,)

    validation_batches = validation_data.fixed_batches(number_of_batches=config.evaluation_batches,batch_size=config.batch_size,context_length=config.context_length,seed=config.seed + 1,)

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(config.seed)

    model = build_model(config).to(device)
    optimizer = build_optimizer(model,learning_rate=config.learning_rate,weight_decay=config.weight_decay,beta1=config.adam_beta1,beta2=config.adam_beta2,epsilon=config.adam_epsilon,)
    scheduler = build_scheduler(optimizer,warmup_steps=config.warmup_steps,total_steps=config.total_steps,minimum_lr_ratio=config.minimum_lr_ratio,)

    start_step = 0
    tokens_processed = 0
    previous_elapsed_seconds = 0.0

    if config.resume is not None:
        checkpoint = load_checkpoint(config.resume,model,optimizer,scheduler,train_generator,)
        validate_resume_config(checkpoint["config"], config)
        start_step = int(checkpoint["step"])
        tokens_processed = int(checkpoint["tokens_processed"])
        previous_elapsed_seconds = float(checkpoint["elapsed_seconds"])
        truncate_jsonl_after_step(log_path, start_step)
        truncate_jsonl_after_step(step_log_path, start_step)

    if config.resume is None or not config_path.exists():
        write_json(config_path, asdict(config))

    model.train()
    run_started = time.perf_counter()
    last_checkpoint_path = None

    prefetcher = BatchPrefetcher(data=train_data,batch_size=config.batch_size,context_length=config.context_length,generator=train_generator,device=device,start_step=start_step,total_steps=config.total_steps,prefetch_batches=config.prefetch_batches,)

    try:
        for step_index in range(start_step, config.total_steps):
            inputs, targets = prefetcher.next_batch()

            optimizer.zero_grad(set_to_none=True)
            learning_rate = optimizer.param_groups[0]["lr"]

            with mixed_precision_context(device, enabled=config.use_bf16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),targets.reshape(-1),)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step_index + 1}")

            loss.backward()
            # Section 5.3: clip after every backward pass, before every optimiser step.
            gradient_norm = clip_gradients(model,max_norm=config.max_grad_norm,)
            optimizer.step()
            scheduler.step()

            completed_step = step_index + 1
            tokens_processed += config.batch_size * config.context_length
            elapsed_seconds = (previous_elapsed_seconds + time.perf_counter() - run_started)
            train_loss = loss.detach().float().item()
            pre_clip_gradient_norm = gradient_norm.detach().float().item()

            step_record = {"step": completed_step,"wall_clock_seconds": elapsed_seconds,"tokens_processed": tokens_processed,"learning_rate": learning_rate,"train_loss": train_loss,"pre_clip_gradient_norm": pre_clip_gradient_norm,}
            append_jsonl(step_log_path, step_record)

            should_evaluate = (completed_step == 1 or completed_step % config.evaluation_interval == 0 or completed_step == config.total_steps)

            if should_evaluate:
                validation_loss = evaluate_model(model,validation_batches,device=device,use_bf16=config.use_bf16,)
                elapsed_seconds = (previous_elapsed_seconds + time.perf_counter() - run_started)

                record = {**step_record,"wall_clock_seconds": elapsed_seconds,"validation_loss": validation_loss,}

                append_jsonl(log_path, record)
                print(json.dumps(record), flush=True)

            should_checkpoint = (completed_step % config.checkpoint_interval == 0 or completed_step == config.total_steps)

            if should_checkpoint:
                last_checkpoint_path = checkpoint_dir / (f"{config.run_name}_step_{completed_step:06d}.pt")

                payload = checkpoint_payload(model=model,optimizer=optimizer,scheduler=scheduler,completed_step=completed_step,tokens_processed=tokens_processed,elapsed_seconds=elapsed_seconds,train_generator=train_generator,config=config,train_generator_state=prefetcher.checkpoint_generator_state(),)
                save_checkpoint(last_checkpoint_path, payload)
    finally:
        prefetcher.close()

    return {"model": model,"optimizer": optimizer,"scheduler": scheduler,"log_path": log_path,"step_log_path": step_log_path,"config_path": config_path,"last_checkpoint_path": last_checkpoint_path,}

def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train the Assignment 1 Transformer language model.")
    parser.add_argument("--train-path", default=TrainingConfig.train_path)
    parser.add_argument("--validation-path",default=TrainingConfig.validation_path,)
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--run-name", default=TrainingConfig.run_name)
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--use-qk-norm",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--use-rmsnorm",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--use-rope",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--ffn-type",choices=("swiglu", "relu"),default="swiglu",)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--evaluation-batches", type=int, default=20)
    parser.add_argument("--validation-reserved-documents",type=int,default=2_000,)
    parser.add_argument("--document-separator-token-id",type=int,default=256,)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3043)
    parser.add_argument("--device",choices=("auto", "cpu", "cuda", "mps"),default="auto",)
    parser.add_argument("--use-bf16",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--resume", default=None)

    return TrainingConfig(**vars(parser.parse_args(argv)))

def main():
    config = parse_args()
    run_training(config)

if __name__ == "__main__":
    main()