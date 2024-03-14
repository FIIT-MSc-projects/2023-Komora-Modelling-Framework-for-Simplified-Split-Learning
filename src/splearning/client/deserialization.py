import os
import sys
import yaml
import torch.nn as nn
import importlib


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

def load_layer(layer):
    model, params = list(layer.items())[0]
    if params is None:
        params = {}
    return LAYER_MAPPING[model](**params)

def create_model(layers):
    model_layers = [load_layer(layer) for layer in layers]
    model = nn.Sequential(*model_layers)
    return model


def load_model_from_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

        if config.get("customCode") is not None:
            module_path = config["customCode"]["modulePath"]
            module_name = config["customCode"]["moduleName"]
            params = config["customCode"]["constructorParams"]


            sys.path.append(module_path)
            module = importlib.import_module(module_name, package=module_path)

            try:
                model = module.model(**params)
            except AttributeError:
                sys.stderr("Model could not be loaded. Function or class named `model` must be provided")
                sys.exit()

        elif config.get("layers") is not None:
            model = create_model(config["layers"])


        if model is None:
            raise ValueError("Model can not be None")
        return model