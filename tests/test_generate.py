from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import sample_next_token


class SamplingTests(unittest.TestCase):
    def test_temperature_zero_is_greedy(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])

        token = sample_next_token(
            logits,
            temperature=0,
        )

        self.assertEqual(token.item(), 1)

    def test_top_k_one_keeps_only_the_best_token(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])

        token = sample_next_token(
            logits,
            temperature=1.0,
            top_k=1,
        )

        self.assertEqual(token.item(), 1)

    def test_top_p_always_keeps_one_token(self):
        logits = torch.tensor([[10.0, 1.0, 0.0]])

        token = sample_next_token(
            logits,
            temperature=1.0,
            top_p=0.0,
        )

        self.assertEqual(token.item(), 0)

    def test_fixed_seed_repeats_the_sample(self):
        logits = torch.zeros((20, 8))

        first_generator = torch.Generator()
        first_generator.manual_seed(3043)

        second_generator = torch.Generator()
        second_generator.manual_seed(3043)

        first = sample_next_token(
            logits,
            generator=first_generator,
        )

        second = sample_next_token(
            logits,
            generator=second_generator,
        )

        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()