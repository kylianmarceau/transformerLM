from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator
import numpy as np

# find the project folder so this script can importtokenizer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import (BPETokenizer,load_merges,load_vocab,save_merges,save_vocab,train_bpe,)

EOT = "<|endoftext|>"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "datasets" / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VALID_PATH = PROJECT_ROOT / "datasets" / "TinyStoriesV2-GPT4-valid.txt"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizers"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "encoded"
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "corpus_encoding.json"
DEFAULT_VOCAB_SIZES = (4_000, 1_000)

# read the file paths and settings supplied on the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Train selected BPE tokenizers and encode TinyStories.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vocab-sizes",type=int,nargs="+",default=list(DEFAULT_VOCAB_SIZES),)
    parser.add_argument("--read-size-mib",type=int,default=8,)
    parser.add_argument("--force",action="store_true",)
    return parser.parse_args()

# keep each tokenizers vocabulary and merges in its own folder
def tokenizer_paths(root: Path, vocab_size: int):
    directory = root / f"vocab_{vocab_size}"
    return directory / "vocab.tsv", directory / "merges.tsv"

# create a fingerprint of a file so the exact tokenizer used is recorded
def sha256(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# train the largest tokenizer once and use prefixes of its learned merges to create the smaller requested tokenizers from the same training run
def train_selected_tokenizers(train_path: Path,tokenizer_root: Path,vocab_sizes: list[int],force: bool,):
    # the starting vocabulary contains 256 bytes and one EOT token
    initial_vocab_size = 256 + 1
    for vocab_size in vocab_sizes:
        if not initial_vocab_size <= vocab_size <= 65_536:
            raise ValueError(f"vocab size {vocab_size} must be between "f"{initial_vocab_size} and 65,536")

    maximum_size = max(vocab_sizes)
    expected_paths = [path for size in vocab_sizes for path in tokenizer_paths(tokenizer_root, size)]

    training_seconds = 0.0

    # reuse existing tokenizer files unless force is supplied
    reused = not force and all(path.is_file() for path in expected_paths)

    if not reused:
        #smaller BPE tokenizers are exact prefixes of the largest merge list
        started = time.perf_counter()
        full_vocab, full_merges = train_bpe(str(train_path), maximum_size, [EOT])
        training_seconds = time.perf_counter() - started

        if len(full_vocab) != maximum_size:
            raise RuntimeError(f"Requested {maximum_size} tokens but training produced "f"only {len(full_vocab)}")

        for vocab_size in vocab_sizes:
            # must keep only the vocabulary entries and merges needed for this size
            vocab = {token_id: full_vocab[token_id] for token_id in range(vocab_size)}
            merge_count = vocab_size - initial_vocab_size
            merges = full_merges[:merge_count]
            vocab_path, merges_path = tokenizer_paths(tokenizer_root, vocab_size)
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            save_vocab(vocab, str(vocab_path))
            save_merges(merges, str(merges_path))

    tokenizers: dict[int, BPETokenizer] = {}
    artifacts: dict[str, object] = {}

    # Load the saved files into the tokenizer objects ready for encoding
    for vocab_size in vocab_sizes:
        vocab_path, merges_path = tokenizer_paths(tokenizer_root, vocab_size)
        vocab = load_vocab(str(vocab_path))
        merges = load_merges(str(merges_path))
        if len(vocab) != vocab_size:
            raise RuntimeError(f"{vocab_path} has {len(vocab)} entries, expected {vocab_size}")
        tokenizers[vocab_size] = BPETokenizer(vocab, merges, [EOT])
        artifacts[str(vocab_size)] = {"vocab_path": str(vocab_path.resolve()),"merges_path": str(merges_path.resolve()),"vocab_sha256": sha256(vocab_path),"merges_sha256": sha256(merges_path),}

    return tokenizers, {"reused_existing": reused,"training_seconds": training_seconds,"trained_maximum_vocab_size": maximum_size,"artifacts": artifacts,}

# read the corpus in chunks that end at a complete EOT boundary, this keeps a document boundary from being split between two separate encoding calls
def iter_boundary_aligned_chunks(path: Path,boundary: str = EOT,read_size: int = 8 * 1024 * 1024,):
    if read_size <= 0:
        raise ValueError("read_size must be positive")
    # buff
    buffer = ""
    with path.open(encoding="utf-8", newline="") as stream:
        while chunk := stream.read(read_size):
            buffer += chunk

            # hold unfinished text until the buffer contains a full boundary
            boundary_start = buffer.rfind(boundary)
            if boundary_start < 0:
                continue
            boundary_end = boundary_start + len(boundary)
            yield buffer[:boundary_end]
            buffer = buffer[boundary_end:]
    if buffer:
        # keep any final text that appears after the last EOT marker
        yield buffer

# Encode a corpus without storing every token ID in memory at once
def encode_to_npy(tokenizer: BPETokenizer,input_path: Path,output_path: Path,read_size: int,force: bool,):
    # reuse an existing valid array unless a fresh run was requested.

    if output_path.exists() and not force:
        array = np.load(output_path, mmap_mode="r")
        if array.dtype != np.uint16 or array.ndim != 1:
            raise RuntimeError(f"Existing array {output_path} is not a one-dimensional uint16 array")
        return {"output_path": str(output_path.resolve()),"token_count": int(array.shape[0]),"encoding_seconds": 0.0,"reused_existing": True,"dtype": str(array.dtype),"input_bytes": input_path.stat().st_size,"output_bytes": output_path.stat().st_size,}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First write raw uint16 values to a temporary file 
    raw_fd, raw_name = tempfile.mkstemp(prefix=f".{output_path.stem}-", suffix=".uint16", dir=output_path.parent)
    os.close(raw_fd)
    raw_path = Path(raw_name)
    npy_path = output_path.with_name(f".{output_path.name}.tmp")
    token_count = 0
    minimum_id: int | None = None
    maximum_id: int | None = None
    started = time.perf_counter()

    try:
        with raw_path.open("wb") as raw_stream:
            for text_chunk in iter_boundary_aligned_chunks(input_path, read_size=read_size):
                token_ids = tokenizer.encode(text_chunk)
                if not token_ids:
                    continue
                chunk_min = min(token_ids)
                chunk_max = max(token_ids)
                # Check before converting so an invalid ID cant wrap around
                if chunk_min < 0 or chunk_max > np.iinfo(np.uint16).max:
                    raise ValueError("Encoded token IDs do not fit in uint16")
                minimum_id = chunk_min if minimum_id is None else min(minimum_id, chunk_min)
                maximum_id = chunk_max if maximum_id is None else max(maximum_id, chunk_max)
                values = np.asarray(token_ids, dtype=np.uint16)
                values.tofile(raw_stream)
                token_count += int(values.size)

        if token_count == 0:
            raise RuntimeError(f"{input_path} encoded to zero tokens")

        # now that the final length is known, create a proper .npy file 
        destination = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint16, shape=(token_count,))
        source = np.memmap(raw_path, mode="r", dtype=np.uint16, shape=(token_count,))
        copy_size = max(1, (64 * 1024 * 1024) // np.dtype(np.uint16).itemsize)
        for start in range(0, token_count, copy_size):
            stop = min(start + copy_size, token_count)
            destination[start:stop] = source[start:stop]
        destination.flush()
        del source, destination

        #  replace only the final output after the complete file is ready
        os.replace(npy_path, output_path)
    finally:
        # remove temporary files
        raw_path.unlink(missing_ok=True)
        npy_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started

    # re open the result 
    check = np.load(output_path, mmap_mode="r")
    if check.dtype != np.uint16 or check.shape != (token_count,):
        raise RuntimeError(f"Verification failed for {output_path}")

    return {"output_path": str(output_path.resolve()),"token_count": token_count,"minimum_token_id": minimum_id,"maximum_token_id": maximum_id,"encoding_seconds": elapsed,"reused_existing": False,"dtype": str(check.dtype),"input_bytes": input_path.stat().st_size,"output_bytes": output_path.stat().st_size,}

# save the timings, tokenizer files, and machine details for later reporting
def write_manifest(path: Path, payload: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # avoid leaving a partly written manifest if the process is interrupted
    os.replace(temporary, path)

# prepare the tokenizers an encode both dataset splits and record the results
def main():
    args = parse_args()
    train_path = args.train.resolve()
    validation_path = args.validation.resolve()
    for path in (train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.read_size_mib <= 0:
        raise ValueError("--read-size-mib must be positive")

    vocab_sizes = list(dict.fromkeys(args.vocab_sizes))
    print(
        "Tokenizer plan: main vocabulary "
        f"{vocab_sizes[0]:,}; comparison vocabularies "
        f"{', '.join(f'{size:,}' for size in vocab_sizes[1:]) or 'none'}"
    )
    tokenizers, training = train_selected_tokenizers(train_path,args.tokenizer_dir.resolve(),vocab_sizes,args.force,)
    print(f"Tokenizer preparation complete in {training['training_seconds']:.2f}s "f"(reused={training['reused_existing']}).")

    encodings: dict[str, object] = {}
    read_size = args.read_size_mib * 1024 * 1024

    # produce training and validation arrays for every vocabulary size
    for vocab_size, tokenizer in tokenizers.items():
        for split, input_path in (("train", train_path),("valid", validation_path),):
            output_path = args.output_dir.resolve() / f"{split}_vocab_{vocab_size}.npy"
            print(f"Encoding {split} with vocabulary {vocab_size:,} -> {output_path}")
            result = encode_to_npy(tokenizer,input_path,output_path,read_size,args.force,)
            encodings[f"{split}_vocab_{vocab_size}"] = result
            print(f"  {result['token_count']:,} tokens, {result['dtype']}, "f"{result['encoding_seconds']:.2f}s "f"(reused={result['reused_existing']})")

    # store the measurements needed for reproducibility and the final report
    payload: dict[str, object] = {

        "section": "3.5",
        "main_vocab_size": vocab_sizes[0],
        "comparison_vocab_sizes": vocab_sizes[1:],
        "special_tokens": [EOT],
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "machine": {"platform": platform.platform(),"python": platform.python_version(),"numpy": np.__version__,},
        "tokenizer_training": training,
        "encodings": encodings,
    }
    write_manifest(args.manifest.resolve(), payload)
    print(f"Manifest written to {args.manifest.resolve()}")

# run the full encoding job 
if __name__ == "__main__":
    main()