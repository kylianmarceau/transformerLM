from __future__ import annotations
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.data import MemoryMappedTokenData
from src.evaluate import (count_target_characters,evaluate_checkpoint,evaluate_non_overlapping_windows,partition_token_range,)
from src.tokenizer import BPETokenizer, END_OF_TEXT

class UniformLanguageModel(torch.nn.Module):
    """A tiny model that gives every token the same probability."""

    def __init__(self, vocab_size: int, context_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = type("Config", (), {"context_length": context_length})()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, token_ids):
        shape = (*token_ids.shape, self.vocab_size)
        return torch.zeros(shape, device=token_ids.device) + self.anchor * 0

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

        self.data_path = Path(self.temporary.name) / "tokens.npy"
        np.save(self.data_path,np.array([ord("a"), 0xC3, 0xA9, 256, ord("b"), ord("c")], dtype=np.uint16),)

        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        vocab[256] = END_OF_TEXT.encode("utf-8")
        self.tokenizer = BPETokenizer(vocab, [], [END_OF_TEXT])

        self.data = MemoryMappedTokenData(self.data_path,reserved_trailing_documents=1,document_separator_token_id=256,)

    def test_partition_keeps_the_reserved_document_out_of_validation(self):
        validation = partition_token_range(self.data, "validation")
        test = partition_token_range(self.data, "test")

        self.assertEqual((validation.start, validation.end), (0, 4))
        self.assertEqual((test.start, test.end), (4, 6))

    def test_uniform_model_has_expected_loss_perplexity_and_bpc(self):
        model = UniformLanguageModel(vocab_size=257, context_length=3)
        token_range = partition_token_range(self.data, "validation")

        result = evaluate_non_overlapping_windows(model=model,data=self.data,token_range=token_range,batch_size=1,device=torch.device("cpu"),use_bf16=False,)
        characters = count_target_characters(token_ids=self.data.tokens,tokenizer=self.tokenizer,input_start=token_range.start,target_end=result["target_end"],include_eot_characters=True,)

        token_count = result["target_token_count"]
        total_nll = result["total_negative_log_likelihood"]
        loss = total_nll / token_count
        perplexity = math.exp(loss)
        bpc = total_nll / (characters["character_count"] * math.log(2))

        self.assertEqual(token_count, 3)
        self.assertEqual(characters["character_count"], 14)  
        self.assertAlmostEqual(loss, math.log(257), places=5)
        self.assertAlmostEqual(perplexity, 257.0, places=3)
        self.assertAlmostEqual(bpc,3 * math.log(257) / (14 * math.log(2)),places=5,)

    def test_eot_characters_can_be_excluded(self):
        result = count_target_characters(token_ids=self.data.tokens,tokenizer=self.tokenizer,input_start=0,target_end=4,include_eot_characters=False,)
        self.assertEqual(result["character_count"], 1)
        self.assertEqual(result["end_of_text_token_count"], 1)
        self.assertEqual(result["trailing_incomplete_utf8_bytes"], 0)

    def test_test_evaluation_requires_explicit_confirmation(self):
        with self.assertRaises(PermissionError):
            evaluate_checkpoint("missing-checkpoint.pt", split="test")

if __name__ == "__main__":
    unittest.main()
