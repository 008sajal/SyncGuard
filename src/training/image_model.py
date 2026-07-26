import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_image_model(
    num_classes: int = 2,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a pretrained ResNet18 binary classifier."""

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, num_classes),
    )

    return model