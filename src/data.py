
"""
training data loader
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

class MemoryMappedTokenData:
    # random batches from a one-dimensional memory-mapped uint16 array

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        self.tokens = np.load(self.path, mmap_mode="r")

        if self.tokens.ndim != 1:
            raise ValueError("Encoded token arrays must be 1 dimensional")

        if self.tokens.dtype != np.uint16:
            raise ValueError("Encoded token arrays must use uint16")

    def sample_batch(self,batch_size: int,context_length: int,generator: torch.Generator,device: str | torch.device = "cpu",):
         # return inputs and one-token-shifted targets

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if context_length <= 0:
            raise ValueError("context_length must be positive")

        maximum_start = len(self.tokens) - context_length

        if maximum_start <= 0:
            raise ValueError("The token array is too short for the requested context length")

        starts = torch.randint(low=0,high=maximum_start,size=(batch_size,),generator=generator,).numpy()

        offsets = np.arange(context_length + 1)
        positions = starts[:, None] + offsets[None, :]

        # Advanced indexing copies only this batch out of the memory map.
        batch = np.asarray(self.tokens[positions], dtype=np.int64)
        batch = torch.from_numpy(batch)

        inputs = batch[:, :-1].to(device=device, non_blocking=True)
        targets = batch[:, 1:].to(device=device, non_blocking=True)

        return inputs, targets

    def fixed_batches(self,number_of_batches: int,batch_size: int,context_length: int,seed: int,):
        # create repeatable CPU validation batches from a fixed seed

        if number_of_batches <= 0:
            raise ValueError("number_of_batches must be positive")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        return [
            self.sample_batch(batch_size=batch_size,context_length=context_length,generator=generator,device="cpu",)
            for _ in range(number_of_batches)
        ]
