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

    def test_utf8_and_whitespace_round_trip(self):
        vocab, _, tokenizer = self.train("hello world\nhello world")
        text = "  Héllø, café!\r\nTabs\tand emoji 🐍" # test for utf8 multi byte coverage
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(vocab[0], b"\x00")
        self.assertEqual(vocab[255], b"\xff")

    # make sure special token is treated as non mergeable bondary 3./3
    def test_special_token_is_a_hard_boundary_and_single_id(self):
        _, merges, tokenizer = self.train(f"abab{EOT}abab", 260)
        special_id = tokenizer.token_to_id[EOT.encode("utf-8")]
        ids = tokenizer.encode(f"ab{EOT}ab")
        self.assertEqual(ids.count(special_id), 1)
        self.assertEqual(tokenizer.decode(ids), f"ab{EOT}ab")
        self.assertNotIn((b"b", b"<"), merges)
        self.assertNotIn((b">", b"a"), merges)
        self.assertFalse(any(EOT.encode("utf-8") in side for pair in merges for side in pair))

    # tie breakign rule when multiple byte pairs have same frequency 
    def test_frequency_ties_choose_lexicographically_greatest_pair(self):
        _, merges, _ = self.train("ab ac", 258)
        self.assertEqual(merges[0], (b"a", b"c"))

    def test_prefix_decode_replaces_incomplete_utf8(self):
        # find a emoji or special character for this 
        _, _, tokenizer = self.train("plain ascii", 257)
        first_byte_of_emoji = tokenizer.token_to_id[b"\xf0"]
        self.assertEqual(tokenizer.decode([first_byte_of_emoji]), "�")

    
