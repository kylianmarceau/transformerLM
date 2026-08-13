# tests for 3.2 

from __future__ import annotations

# imports 
import random
import tempfile
import unittest
from pathlib import Path

# fetch tokenizers
try:
    from src.tokenizer import (BPETokenizer,save_merges,save_vocab,train_bpe,train_bpe_OLD,)

except ModuleNotFoundError:
    from tokenizer import (BPETokenizer,save_merges,save_vocab,train_bpe,train_bpe_OLD,)

EOT = "<|endoftext|>"
PROJECT_ROOT = Path(__file__).resolve().parent

if PROJECT_ROOT.name == "tests":
    PROJECT_ROOT = PROJECT_ROOT.parent

VALIDATION_PATH = (PROJECT_ROOT/ "datasets"/ "TinyStoriesV2-GPT4-valid.txt")


class ByteLevelBPETests(unittest.TestCase):

    # train a small tokenizer on temporary text adn return its vocabulary and merges adn the tokenizer
    def train(self, text: str, vocab_size: int = 272): # 272 default for testing
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        corpus_path = Path(temporary.name) / "corpus.txt"
        corpus_path.write_text(text, encoding="utf-8", newline="")
        vocab, merges = train_bpe(str(corpus_path), vocab_size, [EOT])
        return vocab, merges, BPETokenizer(vocab, merges, [EOT])

    