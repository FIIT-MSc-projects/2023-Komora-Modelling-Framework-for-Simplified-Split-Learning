import torch

def serialize_model(model: torch.nn.Module.__class__, model_path: str):
    torch.save(model(), model_path)

