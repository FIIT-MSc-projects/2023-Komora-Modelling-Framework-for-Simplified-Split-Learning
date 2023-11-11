import os
import yaml
import torch
from torch.nn import *
import torch.nn.functional as F


def serialize_model(model: torch.nn.Module.__class__, model_path: str):
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)    
    torch.save(model(), model_path)

def create_init(code):

    def __init__(self):
        Module.__init__(self)
        exec(code)

    return __init__

def get_layer_init(layer_data):
    code = ""
    layer_function = layer_data[1]['torch_type']
    params = ""
    for param in layer_data[1]['parameters']:
        params += f"{param}={layer_data[1]['parameters'][param]},"
    code += f"self.{layer_data[0]} = {layer_function}({params[:-1]})\n"

    return code

def create_forward_function(code):

    def forward(self, x1):
        exec("x=x1")
        exec(code)
        return eval("x")
        # x = self.fc(x)
        # return x

    return forward

def get_layer_forward(layer_data):
    code = ""
    code += f"x = self.{layer_data[0]}(x)\n"
    if layer_data[1].get('activation_function') is not None:
        code += f"x = F.relu(x)\n"

    return code

def create_model(model_name, layers):

    init_code = ""
    forward_code = ""

    # layers_module_list = ModuleList()

    index = 0

    for layer in layers:
        layer_data = list(layer.items())[0]
        # layer_function = eval(layer_data[1]['torch_type'])
        # layers_module_list.insert(index, layer_function(**layer_data[1]['parameters']))
        index += 1

        init_code += get_layer_init(layer_data)
        forward_code += get_layer_forward(layer_data)


    forward_function = create_forward_function(forward_code)
    init_function = create_init(init_code)


    class_dict = {"__init__": init_function, 'forward': forward_function}
    model = type(f"{model_name}", (Module, ), class_dict)
    
    return model()


def load_model_from_yaml(path, model_name):
    
    with open(path, 'r') as file:
        client_config = yaml.safe_load(file)
        model = create_model(model_name, client_config["layers"])

        return model