"""Evaluate a trained language model with loss, perplexity, and BPC for 6."""

from __future__ import annotations

import argparse
import codecs
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MemoryMappedTokenData
from src.model import TransformerConfig, TransformerLM
from src.tokenizer import BPETokenizer, END_OF_TEXT
from src.train import choose_device, mixed_precision_context, write_json

EvaluationSplit = Literal["validation", "test"]

@dataclass(frozen=True)
class TokenRange:
    # Half open token range belonging to one held-out partition
    start: int
    end: int

    @property
    def token_count(self) -> int:
        return self.end - self.start

def partition_token_range(data: MemoryMappedTokenData,split: EvaluationSplit,):
    """Return validation or reserved-test tokens without copying the memmap."""
    boundary = data.sampling_token_count
    total_tokens = len(data.tokens)

    if split == "validation":
        token_range = TokenRange(0, boundary)
    elif split == "test":
        token_range = TokenRange(boundary, total_tokens)
    else:
        raise ValueError(f"Unsupported evaluation split: {split}")

    if token_range.token_count < 2:
        raise ValueError(f"The {split} split is too short to evaluate")

    return token_range

def model_from_checkpoint(checkpoint_path: str | Path,device: torch.device,):
    #Rebuild the model from the configuration stored in a training checkpoint
    checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=False,)

    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain model state and training config")

    saved_config = checkpoint["config"]
    required_fields = {"vocab_size","context_length","n_layers","d_model","n_heads","d_ff","rope_theta","use_qk_norm","use_rmsnorm","use_rope","ffn_type",}
    missing = required_fields - saved_config.keys()
    if missing:
        raise ValueError(f"Checkpoint config is missing: {sorted(missing)}")

    model_config = TransformerConfig(
        vocab_size=int(saved_config["vocab_size"]),
        context_length=int(saved_config["context_length"]),
        n_layers=int(saved_config["n_layers"]),
        d_model=int(saved_config["d_model"]),
        n_heads=int(saved_config["n_heads"]),
        d_ff=int(saved_config["d_ff"]),
        rope_theta=float(saved_config["rope_theta"]),
        use_qk_norm=bool(saved_config["use_qk_norm"]),
        use_rmsnorm=bool(saved_config["use_rmsnorm"]),
        use_rope=bool(saved_config["use_rope"]),
        ffn_type=str(saved_config["ffn_type"]),
    )
    model = TransformerLM(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint

@torch.no_grad()
def evaluate_non_overlapping_windows(model: TransformerLM,data: MemoryMappedTokenData,token_range: TokenRange,batch_size: int,device: torch.device,use_bf16: bool,max_batches: int | None = None,):
    """Evaluate sequential windows whose target tokens never overlap."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    context_length = model.config.context_length
    available_targets = token_range.token_count - 1
    number_of_windows = available_targets // context_length

    if max_batches is not None:
        number_of_windows = min(number_of_windows, max_batches * batch_size)

    if number_of_windows == 0:
        raise ValueError("The selected split does not contain one complete window")

    total_negative_log_likelihood = 0.0
    total_target_tokens = 0

    for first_window in range(0, number_of_windows, batch_size):
        last_window = min(first_window + batch_size, number_of_windows)
        starts = token_range.start + (np.arange(first_window, last_window, dtype=np.int64) * context_length)
        inputs, targets = data.batch_from_starts(starts, context_length)
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        with mixed_precision_context(device, enabled=use_bf16):
            logits = model(inputs)
            negative_log_likelihood = F.cross_entropy(logits.reshape(-1, logits.size(-1)),targets.reshape(-1),reduction="sum",)

        if not torch.isfinite(negative_log_likelihood):
            raise RuntimeError("Evaluation produced a non-finite loss")

        total_negative_log_likelihood += negative_log_likelihood.float().item()
        total_target_tokens += targets.numel()

    target_start = token_range.start + 1
    target_end = target_start + total_target_tokens

    return {"number_of_windows": number_of_windows,"target_start": target_start,"target_end": target_end,"target_token_count": total_target_tokens,"total_negative_log_likelihood": total_negative_log_likelihood,}

def count_target_characters(token_ids: np.memmap,tokenizer: BPETokenizer,input_start: int,target_end: int,include_eot_characters: bool,chunk_tokens: int = 100_000,):
    # count original UTF 8 characters completed by the evaluated targets
    if not 0 <= input_start < target_end <= len(token_ids):
        raise ValueError("Invalid target token range")

    try:
        eot_id = tokenizer.special_to_id[END_OF_TEXT]
    except KeyError as error:
        raise ValueError("Tokenizer must contain the end-of-text token") from error

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    decoder.decode(tokenizer.vocab[int(token_ids[input_start])], final=False)

    character_count = 0
    eot_count = 0

    for chunk_start in range(input_start + 1, target_end, chunk_tokens):
        chunk_end = min(chunk_start + chunk_tokens, target_end)
        chunk_ids = np.asarray(token_ids[chunk_start:chunk_end])

        try:
            encoded_text = b"".join(tokenizer.vocab[int(token_id)] for token_id in chunk_ids)
        except KeyError as error:
            raise ValueError(f"Encoded array contains unknown token ID {error.args[0]}") from error

        character_count += len(decoder.decode(encoded_text, final=False))
        eot_count += int(np.count_nonzero(chunk_ids == eot_id))

    trailing_utf8_bytes = len(decoder.getstate()[0])

    if not include_eot_characters:
        character_count -= eot_count * len(END_OF_TEXT)

    if character_count <= 0:
        raise ValueError("Evaluated targets contain no countable characters")

    return {"character_count": character_count,"end_of_text_token_count": eot_count,"trailing_incomplete_utf8_bytes": trailing_utf8_bytes,}

def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path | None = None,
    tokenizer_root: str | Path = PROJECT_ROOT / "tokenizers",
    split: EvaluationSplit = "validation",
    reserved_test_documents: int = 2_000,
    document_separator_token_id: int = 256,
    batch_size: int = 32,
    device_name: str = "auto",
    use_bf16: bool = True,
    include_eot_characters: bool = True,
    max_batches: int | None = None,
    confirm_test_once: bool = False,
    output_path: str | Path | None = None,
):
    # evaluate one checkpoint while protecting the reserved test split
    if split == "test" and not confirm_test_once:
        raise PermissionError("Test evaluation is intentionally locked. Use confirm_test_once=True " "only for the final Q18 evaluation.")

    if reserved_test_documents <= 0:
        raise ValueError("reserved_test_documents must be positive")

    device = choose_device(device_name)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    saved_config = checkpoint["config"]

    if data_path is None:
        data_path = saved_config["validation_path"]

    tokenizer_directory = Path(tokenizer_root) / f"vocab_{model.config.vocab_size}"
    vocab_path = tokenizer_directory / "vocab.tsv"
    merges_path = tokenizer_directory / "merges.tsv"
    tokenizer = BPETokenizer.from_files(str(vocab_path),str(merges_path),[END_OF_TEXT],)

    if len(tokenizer.vocab) != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the model")

    data = MemoryMappedTokenData(data_path,reserved_trailing_documents=reserved_test_documents,document_separator_token_id=document_separator_token_id,)
    token_range = partition_token_range(data, split)
    window_result = evaluate_non_overlapping_windows(model=model,data=data,token_range=token_range,batch_size=batch_size,device=device,use_bf16=use_bf16,max_batches=max_batches,)
    character_result = count_target_characters(token_ids=data.tokens,tokenizer=tokenizer,input_start=token_range.start,target_end=int(window_result["target_end"]),include_eot_characters=include_eot_characters,)

    target_token_count = int(window_result["target_token_count"])
    total_negative_log_likelihood = float(window_result["total_negative_log_likelihood"])
    character_count = character_result["character_count"]
    mean_loss = total_negative_log_likelihood / target_token_count
    perplexity = math.exp(mean_loss)
    bits_per_character = total_negative_log_likelihood / (character_count * math.log(2))

    result: dict[str, object] = {
        "split": split,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "data_path": str(Path(data_path).resolve()),
        "vocab_path": str(vocab_path.resolve()),
        "merges_path": str(merges_path.resolve()),
        "device": str(device),
        "precision": ("bf16 autocast" if device.type == "cuda" and use_bf16 else "fp32"),
        "context_length": model.config.context_length,
        "batch_size": batch_size,
        "partition_token_start": token_range.start,
        "partition_token_end": token_range.end,
        "evaluated_target_start": window_result["target_start"],
        "evaluated_target_end": window_result["target_end"],
        "number_of_windows": window_result["number_of_windows"],
        "target_token_count": target_token_count,
        "character_count": character_count,
        "characters_per_token": character_count / target_token_count,
        "end_of_text_characters_included": include_eot_characters,
        "end_of_text_token_count": character_result["end_of_text_token_count"],
        "trailing_incomplete_utf8_bytes": character_result["trailing_incomplete_utf8_bytes"],
        "total_negative_log_likelihood": total_negative_log_likelihood,
        "loss": mean_loss,
        "perplexity": perplexity,
        "bits_per_character": bits_per_character,
        "protocol": (
            "Sequential non-overlapping target windows; characters are assigned "
            "to the target token that completes their UTF-8 encoding."
        ),
    }

    if output_path is not None:
        write_json(Path(output_path), result)

    return result

def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate an Assignment 1 checkpoint with loss, PPL, and BPC.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--tokenizer-root",type=Path,default=PROJECT_ROOT / "tokenizers",)
    parser.add_argument("--split",choices=("validation", "test"),default="validation",)
    parser.add_argument("--reserved-test-documents", type=int, default=2_000)
    parser.add_argument("--document-separator-token-id", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",choices=("auto", "cpu", "cuda", "mps"),default="auto",)
    parser.add_argument("--use-bf16",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--include-eot-characters",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--confirm-test-once", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)

def main():
    args = parse_args()
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        tokenizer_root=args.tokenizer_root,
        split=args.split,
        reserved_test_documents=args.reserved_test_documents,
        document_separator_token_id=args.document_separator_token_id,
        batch_size=args.batch_size,
        device_name=args.device,
        use_bf16=args.use_bf16,
        include_eot_characters=args.include_eot_characters,
        max_batches=args.max_batches,
        confirm_test_once=args.confirm_test_once,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
