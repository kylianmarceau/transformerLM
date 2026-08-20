from src.tokenizer import load_vocab, load_merges
from pathlib import Path
ROOT = Path.cwd()
vocab = load_vocab(ROOT / "tokenizers/vocab_4000/vocab.tsv")
merges = load_merges(ROOT / "tokenizers/vocab_4000/merges.tsv")

token_id, token = max(vocab.items(), key=lambda item: len(item[1]))

print("Longest token ID:", token_id)
print("Length in bytes:", len(token))
print("Token:", repr(token.decode("utf-8", errors="replace")))

print("\nFirst five merges:")
for left, right in merges[:5]:
    print(repr(left.decode("utf-8", "replace")),"+",repr(right.decode("utf-8", "replace")),"->",repr((left + right).decode("utf-8", "replace")),)

print("\nLast five merges:")
for left, right in merges[-5:]:
    print(repr(left.decode("utf-8", "replace")),"+",repr(right.decode("utf-8", "replace")),"->",repr((left + right).decode("utf-8", "replace")),)