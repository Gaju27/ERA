 import torch.nn as nn
 from .custom_net import SimpleCIFARNet


 def create_model(cfg: dict) -> nn.Module:
     name = cfg.get("name", "Custom")
     num_classes = cfg.get("num_classes", 10)
     if name == "Custom":
         return SimpleCIFARNet(num_classes=num_classes)
     raise ValueError(f"Unsupported model: {name}")


