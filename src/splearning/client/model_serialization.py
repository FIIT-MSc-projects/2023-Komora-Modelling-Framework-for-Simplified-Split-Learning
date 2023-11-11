import os
import yaml
import torch.nn as nn
import torch
import collections

LAYER_MAPPING = {
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "Linear": nn.Linear,
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "MaxPool1d": nn.MaxPool1d,
    "MaxPool2d": nn.MaxPool2d,
    "MaxPool3d": nn.MaxPool3d,
    "AvgPool1d": nn.AvgPool1d,
    "AvgPool2d": nn.AvgPool2d,
    "AvgPool3d": nn.AvgPool3d,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "PReLU": nn.PReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "Softmax": nn.Softmax,
    "LogSoftmax": nn.LogSoftmax,
    "Dropout": nn.Dropout,
    "Dropout2d": nn.Dropout2d,
    "Dropout3d": nn.Dropout3d,
    "Embedding": nn.Embedding,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "RNN": nn.RNN,
    "Transformer": nn.Transformer,
    "TransformerEncoder": nn.TransformerEncoder,
    "TransformerDecoder": nn.TransformerDecoder,
}

def create_layer(layer):
    model, params = list(layer.items())[0]
    if params is None:
        params = {}
    return LAYER_MAPPING[model](**params)

def serialize_model(model: nn.Module.__class__, model_path: str):
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)    
    torch.save(model(), model_path)

def create_model(layers):
    model_layers = [create_layer(layer) for layer in layers]
    model = nn.Sequential(*model_layers)
    return model


def load_model_from_yaml(path):
    with open(path, 'r') as file:
        client_config = yaml.safe_load(file)
        model = create_model(client_config["layers"])

        return model