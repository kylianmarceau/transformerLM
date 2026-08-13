# tests for 3.2 

from __future__ import annotations

# imports 
import random
import tempfile
import unittest
# fix error with path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# fetch tokenizers
try:
    from src.tokenizer import (BPETokenizer,save_merges,save_vocab,train_bpe,train_bpe_OLD,)

except ModuleNotFoundError:
    from tokenizer import (BPETokenizer,save_merges,save_vocab,train_bpe,train_bpe_OLD,)

EOT = "<|endoftext|>"

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
        # get special emoji or char too 
        _, _, tokenizer = self.train("plain ascii", 257)
        first_byte_of_emoji = tokenizer.token_to_id[b"\xf0"]
        self.assertEqual(tokenizer.decode([first_byte_of_emoji]), "�")

    def test_encode_iterable_is_lazy_and_preserves_each_item(self):
        _, _, tokenizer = self.train("one two", 260)
        chunks = ["one ", "two", EOT]
        ids = list(tokenizer.encode_iterable(iter(chunks)))
        expected = [token_id for chunk in chunks for token_id in tokenizer.encode(chunk)]
        self.assertEqual(ids, expected)

    def test_from_files_loads_hex_encoded_arbitrary_bytes(self):
        vocab, merges, tokenizer = self.train("banana banana", 264)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        vocab_path = Path(temporary.name) / "vocab.txt"
        merges_path = Path(temporary.name) / "merges.txt"

        save_vocab(vocab, str(vocab_path))
        save_merges(merges, str(merges_path))

        loaded = BPETokenizer.from_files(str(vocab_path),str(merges_path),[EOT],)
        text = f"banana 🍌{EOT}"
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_incremental_training_matches_full_recount_byte_for_byte(self):
        """Section 3.2: optimised bookkeeping must not change BPE."""
        generator = random.Random(3043)
        documents = []
        alphabet = "aaabbbcde '!?é🐍\n"

        for _ in range(80):
            length = generator.randint(1, 90)
            documents.append("".join(generator.choice(alphabet) for _ in range(length)))

        corpus = EOT.join(documents)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        corpus_path = Path(temporary.name) / "corpus.txt"
        corpus_path.write_text(corpus, encoding="utf-8", newline="")

        expected_vocab, expected_merges = train_bpe_OLD(str(corpus_path),340,[EOT],)
        actual_vocab, actual_merges = train_bpe(str(corpus_path),340,[EOT],)

        self.assertEqual(actual_merges, expected_merges)
        self.assertEqual(actual_vocab, expected_vocab)

    def test_incremental_training_handles_overlapping_pairs(self):
        """Repeated symbols must update overlapping pair counts correctly."""
        corpus = EOT.join(["aaaaaaaaaaaa","aaaaaaa","abababababab","banana banana banana",])
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        corpus_path = Path(temporary.name) / "corpus.txt"
        corpus_path.write_text(corpus, encoding="utf-8", newline="")

        expected_vocab, expected_merges = train_bpe_OLD(str(corpus_path),290,[EOT],)
        actual_vocab, actual_merges = train_bpe(str(corpus_path),290,[EOT],)

        self.assertEqual(actual_merges, expected_merges)
        self.assertEqual(actual_vocab, expected_vocab)

    def test_encoding_reuses_cached_pretoken_ids(self):
        _, _, tokenizer = self.train("repeat repeat repeat", 265)
        tokenizer.encode("repeat repeat")
        cached_after_first_encode = dict(tokenizer._cache)
        tokenizer.encode("repeat repeat")
        self.assertEqual(tokenizer._cache, cached_after_first_encode)
        self.assertGreater(len(cached_after_first_encode), 0)

    # 3.3 sanity check 
    # make sure it only rnus if the val set exists
    @unittest.skipUnless(VALIDATION_PATH.exists(),"TinyStories validation dataset is not available",)
    def test_round_trip_on_100_random_validation_documents(self):
        # 3.3 arbitrary validation text must round-trip
        validation_text = VALIDATION_PATH.read_text(encoding="utf-8")
        documents = [document for document in validation_text.split(EOT)if document]

        # dont sample from the final 2000 reserved test documents
        available_documents = documents[:-2_000]
        self.assertGreaterEqual(len(available_documents), 100)

        generator = random.Random(3043)
        sampled_documents = generator.sample(available_documents, 100)
        _, _, tokenizer= self.train(f"small training corpus{EOT}for round-trip testing",272,)

        for document in sampled_documents:
            with self.subTest(document_prefix=document[:40]):
                token_ids = tokenizer.encode(document)
                self.assertEqual(tokenizer.decode(token_ids), document)

    def test_all_token_ids_fit_uint16(self):
        # 3.3 every vocabulary and encoded ID fits uint16
        vocab, _, tokenizer = self.train(f"Bytes, Unicode é and emoji 🐍{EOT}second document",340,)
        token_ids = tokenizer.encode(f"Arbitrary text café 🐍{EOT}with a boundary")

        uint16_max = 65_535
        self.assertTrue(all(0 <= token_id <=uint16_max for token_id in vocab))
        
        self.assertTrue(all(0 <= token_id <= uint16_max for token_id in token_ids))

        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.txt"
            corpus_path.write_text("small corpus", encoding="utf-8")

            with self.assertRaises(ValueError):
                train_bpe(str(corpus_path),uint16_max + 2,[EOT],)

    # test training twice on same text learns same merges and vocabs 
    def test_training_is_deterministic(self):
        # 3.3 repeated runs must learn identical merges 
        corpus = EOT.join(["the quick brown fox jumps over the lazy dog", "the quick blue bird flies over the quiet lake", "Unicode: Héllø, café! 🐍",])

        first_vocab, first_merges, _ = self.train(corpus, 340)
        second_vocab, second_merges, _ = self.train(corpus, 340)

        self.assertEqual(second_merges, first_merges)
        self.assertEqual(second_vocab, first_vocab)



# test


if __name__ == "__main__":
    unittest.main()
