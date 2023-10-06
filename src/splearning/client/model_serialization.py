import os
import torch

def serialize_model(model: torch.nn.Module.__class__, model_path: str):
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)    
    torch.save(model(), model_path)

