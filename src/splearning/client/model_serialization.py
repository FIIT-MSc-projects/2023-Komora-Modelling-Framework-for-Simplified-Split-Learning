import os
import torch

def serialize_model(model: torch.nn.Module.__class__, model_path: str):
    if not os.path.isdir(os.path.dirname(model_path)):
        os.mkdir(os.path.dirname(model_path))    
    torch.save(model(), model_path)

