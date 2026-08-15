from __future__ import annotations
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
