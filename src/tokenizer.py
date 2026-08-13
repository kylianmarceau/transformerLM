# imports

from __future__ import annotations
from collections import Counter
import heapq
from typing import Iterable, Iterator

import regex

END_OF_TEXT = "<|endoftext|>"

# gpt2 pre-tokenizer regex from appendix A
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

GPT2_pretoken_regex = regex.compile(GPT2_PAT)

def validate_special_tokens(special_tokens: list[str] | None, ):
    # validate and copy the configured special token list
    tokens = list(special_tokens or [])

    if any(token == "" for token in tokens):
        raise ValueError("Special tokens cannot be empty")

    if len(tokens) != len(set(tokens)):
        raise ValueError("Special tokens must be unique")

    return tokens

def split_on_special_tokens(text: str, special_tokens: list[str], keep: bool,):
    # split at the special tokens, toptionallu retaining them 
    if not special_tokens:
        return[text] 

    #macth longer special tokens first 
    ordered = sorted(special_tokens ,key=lambda token: (-len(token), token),)
    alternatives = "|".join(regex.escape(token) for token in ordered)

    if keep:
        pattern = f"({alternatives})"
    else:
        pattern = f"(?:{alternatives})"

    return regex.split(pattern, text)

def pretokenize(text: str, ):
    # convert gpt2 pretokens into tuples of individual bytes
    pretokens = []
    for match in GPT2_pretoken_regex.finditer(text):
        raw_bytes = match.group(0).encode("utf-8")
        symbols = tuple(bytes([byte_value])for byte_value in raw_bytes) 
        pretokens.append(symbols)

    return pretokens

def count_pretokens(text: str, special_tokens: list[str], ):
    # count byte pre tokens wihtout counting the special tokens
    counts: Counter = Counter()

    for chunk in split_on_special_tokens(text, special_tokens, keep=False,):
        if not chunk:
            continue

        counts.update(pretokenize(chunk))

    return counts

def count_pairs(word_freqs: dict[tuple[bytes, ...], int], ): #CHANGE signature so refers bytes
    """
    Count how often each adjacent pair of symbols occurs, weighted by pre-token frequency.
    params:
        word_freqs: dict mapping a pre-token (tuple of symbol strings) -> its corpus count
    returns:
        Counter mapping (left_symbol, right_symbol) -> total count
    """

    # count adjacebnt byte symbol paris weighted by frequency
    pair_counts: Counter = Counter()
    for symbols, frequency in word_freqs.items():
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += frequency
    return pair_counts

class PairPriority:
    # heap entry which places the most frequent, lexicographically greatest pair first

    def __init__(self,pair: tuple[bytes, bytes],count: int,):
        self.pair = pair
        self.count = count

    def __lt__(self,other):
        if self.count != other.count:
            return self.count > other.count

        return self.pair > other.pair

def update_pair_count(pair_counts: dict[tuple[bytes, bytes], int],pair_heap: list[PairPriority],pair: tuple[bytes, bytes],change: int,):
    # update one pair count and add its new value to the priority heap
    new_count = pair_counts.get(pair, 0) + change

    if new_count < 0:
        raise ValueError(f"Pair count became negative for {pair!r}")

    if new_count == 0:
        pair_counts.pop(pair, None)
    else:
        pair_counts[pair] = new_count
        heapq.heappush(pair_heap,PairPriority(pair,new_count,),)

def best_pair_from_heap(pair_counts: dict[tuple[bytes, bytes], int],pair_heap: list[PairPriority],):
    # discard old heap entries until the stored count matches the current count
    while pair_heap:
        entry = heapq.heappop(pair_heap)

        if pair_counts.get(entry.pair) == entry.count:
            return entry.pair

    return None

def build_pair_indexes(words: list[tuple[bytes, ...]],word_freqs: list[int],):
    # count all pairs once and record which pre-tokens contain them
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}

    for word_id, symbols in enumerate(words):
        local_pair_counts = Counter(zip(symbols, symbols[1:]))
        frequency = word_freqs[word_id]

        for pair, occurrences in local_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * frequency
            pair_to_words.setdefault(pair, set()).add(word_id)

    pair_heap = [PairPriority(pair,count,) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    return pair_counts, pair_to_words, pair_heap

def merge_word(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes], ): # CHANGE to bytesz
    """
    Replace every occurrence of `pair` in `symbols` with the single merged symbol.
    E.g. merge_word(("n", "e", "w", "e", "s", "t"), ("s", "t")) -> ("n", "e", "w", "e", "st")
    params:
        symbols: tuple of symbol strings representing one pre-token
        pair:    (left, right) tuple of symbols to merge
    returns:
        new tuple of symbols
    """
    merged = []
    index = 0
    while index < len(symbols):
        # If the pair starts here, append the concatenated symbol and skip forward by 2.
        if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index+ 1])
            index += 2
        # Otherwise keep the current symbol and advance by 1.
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)

# -- 3.2 --
def merge_pair_incrementally(best_pair: tuple[bytes, bytes],words: list[tuple[bytes, ...]],word_freqs: list[int],pair_counts: dict[tuple[bytes, bytes], int],pair_to_words: dict[tuple[bytes, bytes], set[int]],pair_heap: list[PairPriority],):
    # update only the pre-tokens which contain the selected pair
    affected_word_ids = list(pair_to_words.get(best_pair, set()))

    if not affected_word_ids:
        raise ValueError("Pair index does not match pair counts")

    pair_changes: Counter = Counter()

    for word_id in affected_word_ids:
        symbols = words[word_id]
        frequency = word_freqs[word_id]

        # remove the old pair counts for this pre-token
        old_pair_counts = Counter(zip(symbols, symbols[1:]))

        for pair, occurrences in old_pair_counts.items():
            pair_changes[pair] -= occurrences * frequency
            indexed_words = pair_to_words.get(pair)

            if indexed_words is not None:
                indexed_words.discard(word_id)

                if not indexed_words:
                    del pair_to_words[pair]

        # apply the merge and add the new pair counts
        merged_symbols = merge_word(symbols,best_pair,)
        words[word_id] = merged_symbols
        new_pair_counts = Counter(zip(merged_symbols, merged_symbols[1:]))

        for pair, occurrences in new_pair_counts.items():
            pair_changes[pair] += occurrences * frequency
            pair_to_words.setdefault(pair, set()).add(word_id)

    for pair, change in pair_changes.items():
        if change:
            update_pair_count(pair_counts,pair_heap,pair,change,)

def train_bpe_OLD(input_path: str,vocab_size: int,special_tokens: list[str],) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    """Train BPE by recounting every pair after every merge"""
    special_tokens = validate_special_tokens(special_tokens)

    if vocab_size > 65_536:
        raise ValueError(
            "vocab_size must not exceed 65,536 "
            "when using uint16 token IDs")

    with open(input_path,encoding="utf-8",newline="",) as stream:
        text = stream.read()

    vocab: dict[int, bytes] = {byte_value: bytes([byte_value])for byte_value in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    number_of_merges = vocab_size - len(vocab)

    if number_of_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is too small; "f"{len(vocab)} initial tokens are required")

    word_freqs = dict(count_pretokens(text,special_tokens,))
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(number_of_merges):
        pair_counts = count_pairs(word_freqs)

        if not pair_counts:
            break

        best_pair = max(pair_counts.items(),key=lambda item: (item[1], item[0]),)[0]
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        new_word_freqs: Counter = Counter()

        for symbols, frequency in word_freqs.items():
            merged_symbols = merge_word(symbols,best_pair,)
            new_word_freqs[merged_symbols] += frequency

        word_freqs = new_word_freqs

    return vocab, merges

def train_bpe(input_path: str,vocab_size: int,special_tokens: list[str],) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    """Train a byte level BPE tokenizer using incremental pair counting

    Returns:
        vocab: token ID -> token bytes
        merges: l earned byte pairs in learning order
    """
    special_tokens = validate_special_tokens(
        special_tokens
    )

    if vocab_size > 65_536:
        raise ValueError(
            "vocab_size must not exceed 65,536 "
            "when using uint16 token IDs"
        )

    with open(
        input_path,
        encoding="utf-8",
        newline="",
    ) as stream:
        text = stream.read()

    # IDs 0 - 255 are the corresponding individual bytes.
    vocab: dict[int, bytes] = {byte_value: bytes([byte_value])for byte_value in range(256)}

    # Special tokens follow the base byte vocabulary.
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    number_of_merges = vocab_size - len(vocab)

    if number_of_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is too small; "f"{len(vocab)} initial tokens are required")

    pretoken_counts = count_pretokens(text,special_tokens,)

    # Store pre-tokens by integer ID so they can be updated in place.
    words = list(pretoken_counts)
    word_freqs = [pretoken_counts[word] for word in words]

    # Count pairs once, and record which pre-tokens contain each pair.
    pair_counts, pair_to_words, pair_heap = build_pair_indexes(words,word_freqs,)

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(number_of_merges):
        if not pair_counts:
            break

        best_pair = best_pair_from_heap(pair_counts,pair_heap,)

        if best_pair is None:
            break

        vocab[len(vocab)] = (best_pair[0] + best_pair[1])
        merges.append(best_pair)

        merge_pair_incrementally(best_pair,words,word_freqs,pair_counts,pair_to_words,pair_heap,)

        # Rebuild occasionally so stale priority entries do not waste memory.
        if len(pair_heap) > max(1_000, 4 * len(pair_counts)):
            pair_heap = [PairPriority(pair,count,) for pair, count in pair_counts.items()]
            heapq.heapify(pair_heap)

    return vocab, merges

def save_vocab(vocab: dict[int, bytes],path: str,):
    # Save token IDs and byte values using hexadecimal 
    with open(path, "w", encoding="utf-8") as stream:
        for token_id in sorted(vocab):
            stream.write(f"{token_id}\t{vocab[token_id].hex()}\n")


def save_merges(merges: list[tuple[bytes, bytes]],path: str,):
    # save ordered byte-pair merges using hexadecimal
    with open(path, "w", encoding="utf-8") as stream:
        for left, right in merges:
            stream.write(f"{left.hex()}\t{right.hex()}\n")

def load_vocab(path: str):
    # Load the hexadecimal vocabulary format
    vocab: dict[int, bytes] = {}

    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            fields = line.split("\t", 1)

            if len(fields) != 2:
                raise ValueError(f"Invalid vocabulary line {line_number}: {line!r}")

            token_id_text, token_hex = fields
            vocab[int(token_id_text)] = bytes.fromhex(token_hex)

    return vocab

def load_merges(path: str,):
    # load hexadecimal byte-pair merges in learned order 
    merges: list[tuple[bytes, bytes]] = []

    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            fields = line.split("\t")

            if len(fields) != 2:
                raise ValueError(f"Invalid merge line {line_number}: {line!r}")

            left_hex, right_hex = fields
            merges.append((bytes.fromhex(left_hex),bytes.fromhex(right_hex),))

    return merges

class BPETokenizer:
    """Byte level BPE tokenizer """

    def __init__(self,vocab: dict[int, bytes],merges: list[tuple[bytes, bytes]],special_tokens: list[str] | None = None,):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = validate_special_tokens(special_tokens)

        # Validate the required base vocabulary.
        for byte_value in range(256):
            if self.vocab.get(byte_value) != bytes([byte_value]):
                raise ValueError(
                    "Vocabulary IDs 0–255 must contain "
                    "their corresponding byte values"
                )

        self.token_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.special_to_id: dict[str, int] = {}

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes not in self.token_to_id:
                raise ValueError(
                    f"Special token {special_token!r} "
                    "is missing from the vocabulary"
                )

            self.special_to_id[special_token] = (self.token_to_id[special_bytes])

        self._cache: dict[tuple[bytes, ...],list[int],] = {}

    @classmethod
    def from_files(cls,vocab_path: str,merges_path: str,special_tokens: list[str] | None = None,):
        vocab = load_vocab(vocab_path)
        merges = load_merges(merges_path)

        return cls(vocab,merges,special_tokens=special_tokens,)

    def _apply_merges(self,symbols: tuple[bytes, ...],):
        # Apply the earliest-learned applicable merge 
        symbols = list(symbols)

        while len(symbols) > 1:
            best_rank = None
            best_index = None

            for index, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_index = index

            if best_index is None:
                break

            symbols[best_index : best_index + 2] = [symbols[best_index]+ symbols[best_index + 1]]

        return symbols

    def _encode_pretoken(self,pretoken: tuple[bytes, ...],):
        if pretoken not in self._cache:
            merged_symbols = self._apply_merges(pretoken)

            self._cache[pretoken] = [self.token_to_id[symbol] for symbol in merged_symbols]

        return self._cache[pretoken]

    def encode(self, text: str) -> list[int]:
        # encode text into byte BPE token IDs 
        token_ids: list[int] = []

        for chunk in split_on_special_tokens(text,self.special_tokens,keep=True,):
            if chunk in self.special_to_id:
                token_ids.append(
                    self.special_to_id[chunk]
                )
            elif chunk:
                for pretoken in pretokenize(chunk):
                    token_ids.extend(
                        self._encode_pretoken(pretoken)
                    )


        return token_ids

    def encode_iterable(self,iterable: Iterable[str],) -> Iterator[int]:
        
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        #Decode IDs without crashing out on incomplete UTF 8 
        pieces = []

        for token_id in ids:
            integer_id = int(token_id)

            if integer_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {integer_id}")

            pieces.append(self.vocab[integer_id])

        raw_bytes = b"".join(pieces)

        return raw_bytes.decode("utf-8",errors="replace",)