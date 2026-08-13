
"""
Choosing a vocabulary size
Train BPE at a range of vocabulary
sizes (at least 1,000 / 2,000 / 4,000 / 8,000 / 16,000) on the validation file or a subsample of the training
file, and measure the compression ratio for each.
Note that you can track the corpus token count incrementally during training almost for free — every
merge reduces it by exactly the count of the merged pair.

must produce curve and probably a csv 

"""

from __future__ import annotations
import argparse
import csv 
import sys 
from pathlib import Path
import time
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import (BPETokenizer,save_merges,save_vocab,train_bpe,)


EOT = "<|endoftext|>"
DEFAULT_VOCAB_SIZES = (1_000, 2_000, 4_000, 8_000, 16_000)
DEFAULT_VALIDATION_PATH = (PROJECT_ROOT / "datasets" / "TinyStoriesV2-GPT4-valid.txt")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "vocab_study"

# read args from the command line for the vocab choice to test 
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Assignment 1 Section 3.4 vocabulary study.")
    parser.add_argument("--input",type=Path,default=DEFAULT_VALIDATION_PATH,)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR,)
    parser.add_argument("--vocab-sizes",type=int,nargs="+",default=list(DEFAULT_VOCAB_SIZES),)
    parser.add_argument("--reserved-test-documents",type=int,default=2_000,)
    parser.add_argument("--max-documents",type=int,default=None,)
    parser.add_argument("--d-model",type=int,default=512,)
    return parser.parse_args()


def load_study_corpus(input_path: Path,reserved_test_documents: int,max_documents: int | None,):
    #Return eligible validation documents joined by hard EOT boundaries
    if reserved_test_documents < 0:
        raise ValueError("res test docs cant be negative")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"Validation file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    documents = [document for document in text.split(EOT) if document]

    if reserved_test_documents >= len(documents):
        raise ValueError("reserved_test_documents must be smaller than the number " f"of documents ({len(documents)})")

    eligible = (documents[:-reserved_test_documents] if reserved_test_documents else documents)
    if max_documents is not None:
        eligible = eligible[:max_documents]

    return EOT.join(eligible), len(eligible)


def tokenizer_paths(output_dir: Path, vocab_size: int):
    tokenizer_dir = output_dir / "tokenizers" / f"vocab_{vocab_size}"
    return tokenizer_dir / "vocab.tsv", tokenizer_dir / "merges.tsv" #saves

def display_path(path: Path):
    #Prefer a project relative artifact path, otherwise use its full path 
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train_and_measure(corpus_path: Path,corpus_text: str,document_count: int,requested_vocab_size: int,d_model: int,output_dir: Path,):
    #Train one tokenizer then encode the corpus and return its measurements
    if not 257 <= requested_vocab_size <= 65_536:
        raise ValueError(f"vocab size {requested_vocab_size} must be between 257 and 65,536")

    train_started = time.perf_counter()
    vocab, merges = train_bpe(str(corpus_path),requested_vocab_size,[EOT],)
    training_seconds = time.perf_counter() - train_started

    tokenizer = BPETokenizer(vocab, merges, [EOT])
    encode_started = time.perf_counter()
    token_ids = tokenizer.encode(corpus_text)
    encoding_seconds = time.perf_counter() - encode_started

    if not token_ids:
        raise RuntimeError("The study corpus encoded to zero tokens")

    vocab_path, merges_path = tokenizer_paths(output_dir, requested_vocab_size)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, str(vocab_path))
    save_merges(merges, str(merges_path))

    byte_count = len(corpus_text.encode("utf-8"))
    character_count = len(corpus_text)
    token_count = len(token_ids)

    return {"requested_vocab_size": requested_vocab_size,"actual_vocab_size": len(vocab),"merge_count": len(merges),"document_count": document_count,"character_count": character_count,"byte_count": byte_count,"token_count": token_count,"bytes_per_token": byte_count / token_count,"characters_per_token": character_count / token_count,"training_seconds": training_seconds,"encoding_seconds": encoding_seconds,"embedding_parameters": len(vocab) * d_model,"lm_head_parameters": len(vocab) * d_model,"vocab_dependent_parameters": 2 * len(vocab) * d_model,"vocab_path": display_path(vocab_path),"merges_path": display_path(merges_path),}


