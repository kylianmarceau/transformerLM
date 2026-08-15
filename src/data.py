"""training data loader"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

class MemoryMappedTokenData:
    """Sample batches from a one-dimensional memory-mapped uint16 array."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        # mmap_mode keeps the full token array on disk instead of loading into RAM individual batches are copied only when requested
        self.tokens = np.load(self.path, mmap_mode="r")

        if self.tokens.ndim != 1:
            raise ValueError("Encoded token arrays must be 1 dimensional")

        if self.tokens.dtype != np.uint16:
            raise ValueError("Encoded token arrays must use uint16")

    def sample_starts(self,batch_size: int,context_length: int,generator: torch.Generator,):
        #choose the starting token for each sequence in a batch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if context_length <= 0:
            raise ValueError("context_length must be positive")

        maximum_start = len(self.tokens) - context_length

        if maximum_start <= 0:
            raise ValueError("The token array is too short for the requested context length")

        return torch.randint(low=0,high=maximum_start,size=(batch_size,),generator=generator,).numpy()

    def batch_from_starts(self,starts: np.ndarray,context_length: int,):
        #copy one batch from the memory map into CPU tensors

        offsets = np.arange(context_length + 1)
        positions = starts[:, None] + offsets[None, :]

        # Advanced indexing copies only this batch, not the full token array.
        batch = np.asarray(self.tokens[positions], dtype=np.int64)
        batch = torch.from_numpy(batch)

        return batch[:, :-1], batch[:, 1:]

    def sample_batch(self,batch_size: int,context_length: int,generator: torch.Generator,device: str | torch.device = "cpu",):
        #return inputs and their one-token-shifted targets

        starts = self.sample_starts(batch_size=batch_size,context_length=context_length,generator=generator,)
        inputs, targets = self.batch_from_starts(starts, context_length)

        return (inputs.to(device=device, non_blocking=True),targets.to(device=device, non_blocking=True),)

    def fixed_batches(self,number_of_batches: int,batch_size: int,context_length: int,seed: int,):
        """Create repeatable CPU validation batches from a fixed seed"""

        if number_of_batches <= 0:
            raise ValueError("number_of_batches must be positive")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        return [self.sample_batch(batch_size=batch_size,context_length=context_length,generator=generator,device="cpu",) for _ in range(number_of_batches)]

class BatchPrefetcher:
    """Prepare upcoming training batches on a background thread."""

    def __init__(self,data: MemoryMappedTokenData,batch_size: int,context_length: int,generator: torch.Generator,device: str | torch.device,start_step: int,total_steps: int,prefetch_batches: int = 2,):
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")

        if not 0 <= start_step <= total_steps:
            raise ValueError("start_step must be between 0 and total_steps")

        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.generator = generator
        self.device = torch.device(device)
        self.total_steps = total_steps
        self.prefetch_batches = prefetch_batches
        self.next_step_to_schedule = start_step

        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix="batch-prefetch",)
        self.pending: deque[
            tuple[
                torch.Tensor,
                Future[tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = deque()

        # Fill the queue before training starts. Further reads happen in the
        # background while the model works on the current batch.
        self._fill_queue()

    def _fill_queue(self) -> None:
        while (
            len(self.pending) < self.prefetch_batches
            and self.next_step_to_schedule < self.total_steps
        ):
            # Store the state before selecting this batch. If a checkpoint is
            # saved while it is prefetched, resuming can recreate it exactly.
            generator_state = self.generator.get_state().clone()
            starts = self.data.sample_starts(batch_size=self.batch_size,context_length=self.context_length,generator=self.generator,)
            future = self.executor.submit(self.data.batch_from_starts,starts,self.context_length,)

            self.pending.append((generator_state, future))
            self.next_step_to_schedule += 1

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # return the oldest prepared batch and begin preparing another

        if not self.pending:
            raise StopIteration("No training batches remain")

        _, future = self.pending.popleft()
        inputs, targets = future.result()

        # Refill before returning so this disk read overlaps model computation.
        self._fill_queue()

        return (inputs.to(device=self.device, non_blocking=True),targets.to(device=self.device, non_blocking=True),)

    def checkpoint_generator_state(self) -> torch.Tensor:
        # retun the RNG state before the oldest unused batch

        if self.pending:
            return self.pending[0][0].clone()

        return self.generator.get_state().clone()

    def close(self):
        self.executor.shutdown(wait=True)
