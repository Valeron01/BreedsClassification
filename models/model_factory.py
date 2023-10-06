import torch


def get_model_by_config(model_config: dict):
    model_name = model_config["model_name"]
    if model_name == "SimplePlaneNetwork":
        model = None
        return model

    raise NotImplementedError(f"Model with name {model_name} is not implemented!")
