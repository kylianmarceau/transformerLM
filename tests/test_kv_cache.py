from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TransformerConfig, TransformerLM

class KVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3043)
        config = TransformerConfig(vocab_size=64,context_length=128,n_layers=2,d_model=32,n_heads=4,d_ff=64,)
        self.model = TransformerLM(config).eval()
        self.prompt = torch.randint(0, config.vocab_size, (1, 8))

    def test_cache_length_and_reset(self):
        self.model.reset_cache()

        with torch.no_grad():
            self.model(self.prompt, use_cache=True)

        self.assertEqual(self.model.cache_length, self.prompt.size(1))

        next_token = torch.randint(0, 64, (1, 1))

        with torch.no_grad():
            self.model(next_token, use_cache=True)

        self.assertEqual(self.model.cache_length, self.prompt.size(1) + 1)

        self.model.reset_cache()
        self.assertEqual(self.model.cache_length, 0)

    def test_cached_and_uncached_generation_match(self):
        number_of_new_tokens = 100

        # Generate by rerunning the complete sequence each time.
        uncached_tokens = self.prompt.clone()
        uncached_logits = []

        with torch.no_grad():
            for _ in range(number_of_new_tokens):
                logits = self.model(uncached_tokens,use_cache=False,)[:, -1]
                uncached_logits.append(logits)
                next_token = logits.argmax(dim=-1, keepdim=True)
                uncached_tokens = torch.cat((uncached_tokens, next_token),dim=1,)

        # Prefill once, then pass only the newest token to the model.
        self.model.reset_cache()
        cached_tokens = self.prompt.clone()
        cached_logits = []

        with torch.no_grad():
            logits = self.model(cached_tokens,use_cache=True,)[:, -1]

            for step in range(number_of_new_tokens):
                cached_logits.append(logits)
                next_token = logits.argmax(dim=-1, keepdim=True)
                cached_tokens = torch.cat((cached_tokens, next_token),dim=1,)

                if step < number_of_new_tokens - 1:
                    logits = self.model(next_token,use_cache=True,)[:, -1]

        uncached_logits = torch.stack(uncached_logits)
        cached_logits = torch.stack(cached_logits)

        maximum_difference = (uncached_logits - cached_logits).abs().max().item()

        generations_identical = torch.equal(cached_tokens,uncached_tokens,)

        print("\nMaximum absolute logit difference:", maximum_difference)
        print("Greedy generations identical:", generations_identical)

        self.assertTrue(generations_identical)
        self.assertLess(maximum_difference, 1e-4)


if __name__ == "__main__":
    unittest.main()
