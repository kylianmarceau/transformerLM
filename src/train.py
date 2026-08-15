from __future__ import annotations
import math
import torch 

def build_optimizer(model, learning_rate: float, weight_decay: float = 0.1):
    """build the adam optimzer for 5.1"""

    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    if weight_decay < 0:
        raise ValueError("learning rate cannot be negative")

    decay_parameters=[]
    no_decay_parameters=[]

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [{"params": decay_parameters, "weight_decay": weight_decay,},{"params": no_decay_parameters, "weight_decay": 0.0,},]

    return torch.optim.AdamW(parameter_groups, lr=learning_rate, betas = (0.9, 0.95), eps=1e-8,)


def build_scheduler(optimizer,warmup_steps: int,total_steps: int,minimum_lr_ratio: float = 0.1,):
    # Build the linear warmup and cosine decay scheduler for 5.2
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")

    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")

    if not 0 < minimum_lr_ratio <= 1:
        raise ValueError("minimum_lr_ratio must be between 0 and 1")

    decay_steps = total_steps - warmup_steps

    def learning_rate_multiplier(step: int):
        # the first optimiser step starts above zero and the last warmup step reaches the configured maximum learning rate
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        # cosine period is tied to this runs actual number of steps
        if decay_steps == 1:
            return minimum_lr_ratio

        progress = (step - warmup_steps)/(decay_steps-1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=learning_rate_multiplier,)