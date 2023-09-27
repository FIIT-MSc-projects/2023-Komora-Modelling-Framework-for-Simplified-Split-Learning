import torch


def deserialize_model(model_path: str) -> torch.nn.Module: 
    return torch.load(model_path)