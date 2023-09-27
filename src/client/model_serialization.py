import torch

def serialize_model(model: torch.nn.Module.__class__, model_path: str):
    torch.save(model(), model_path)

def deserialize_model(model_path: str) -> torch.nn.Module: 
    return torch.load(model_path)
