import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=200, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
