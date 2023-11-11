import os
from torchinfo import summary
import yaml
import torch
from torch.nn import *
import torch.nn.functional as F
import collections

class ModelBuilder():
    def __init__(self):
        self.mapping = collections.ChainMap({
            "Conv2d": Conv2d,
            "MaxPool2d": MaxPool2d,
            "Linear": Linear,
            "ReLU": ReLU
        })

    def build_model(self, model, **params):
        return self.mapping[model](**params)

def serialize_model(model: torch.nn.Module.__class__, model_path: str):
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)    
    torch.save(model(), model_path)

def create_model(layers):
    model_layers = []
    builder = ModelBuilder()

    for layer in layers:

        name, kwargs = list(layer.items())[0]
        if kwargs is None:
            kwargs = {}

        
        model_layers.append(builder.build_model(name, **kwargs))
        print(builder.build_model(name, **kwargs))

    model = Sequential(*model_layers)
    print(model)
    
    return model


def load_model_from_yaml(path):
    
    with open(path, 'r') as file:
        client_config = yaml.safe_load(file)
        model = create_model(client_config["layers"])

        return model