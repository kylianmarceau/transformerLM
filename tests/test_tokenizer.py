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


