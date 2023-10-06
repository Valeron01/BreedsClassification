import typing

import torch


def get_optimizer_by_config(parameters: typing.Iterable[torch.nn.Parameter], optimizer_config: dict):
    optimizer_name = optimizer_config["optimizer_name"]
    lr = optimizer_config["lr"]

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(parameters, lr)
        return optimizer

    raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented!")
