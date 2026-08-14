from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ReLUFFN, RMSNorm, TransformerConfig, TransformerLM


class TransformerModelTests(unittest.TestCase):
    def small_config(self, **changes):
        values = {"vocab_size": 100,"context_length": 16,"n_layers": 2,"d_model": 32,"n_heads": 4,"d_ff": 64,}
        values.update(changes)
        return TransformerConfig(**values)

    def test_base_configuration_matches_section_4_1(self):
        config = TransformerConfig()
        self.assertEqual(config.vocab_size, 4000)
        self.assertEqual(config.context_length, 256)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.d_ff, 1344)
        self.assertEqual(config.rope_theta, 10_000.0)
        self.assertTrue(config.use_qk_norm)

    def test_forward_shape_and_causal_mask(self):
        torch.manual_seed(3043)
        model = TransformerLM(self.small_config()).eval()
        token_ids = torch.randint(0, 100, (2, 8))

        with torch.no_grad():
            logits = model(token_ids)

            changed = token_ids.clone()
            changed[:, 5:] = torch.randint(0, 100, changed[:, 5:].shape)
            changed_logits = model(changed)

        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits[:, :5], changed_logits[:, :5])

    def test_ablation_options_change_the_expected_modules(self):
        no_norm = TransformerLM(self.small_config(use_rmsnorm=False))
        no_rope = TransformerLM(self.small_config(use_rope=False))
        relu = TransformerLM(self.small_config(ffn_type="relu"))

        self.assertFalse(any(isinstance(module, RMSNorm) for module in no_norm.modules()))
        self.assertTrue(all(not layer.attn.use_rope for layer in no_rope.layers))
        self.assertTrue(all(isinstance(layer.ffn, ReLUFFN) for layer in relu.layers))

        self.assertIsInstance(no_norm.final_norm, nn.Identity)

if __name__ == "__main__":
    unittest.main()
