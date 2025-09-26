from models import model_v1, model_v2, model_v3
from imports import torch

class ModelStrategy:
    def get_model(self, device):
        raise NotImplementedError

class Model1Strategy(ModelStrategy):
    def get_model(self, device):
        return model_v1.Net().to(device)

class Model2Strategy(ModelStrategy):
    def get_model(self, device):
        return model_v2.Net().to(device)

class Model3Strategy(ModelStrategy):
    def get_model(self, device):
        return model_v3.Net().to(device)

class ModelContext:
    def __init__(self, strategy: ModelStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelStrategy):
        self._strategy = strategy

    def get_model(self, device):
        return self._strategy.get_model(device)

def get_strategy(version):
    if version == '1':
        return Model1Strategy()
    elif version == '2':
        return Model2Strategy()
    elif version == '3':
        return Model3Strategy()
    else:
        raise ValueError('Unknown model version')
