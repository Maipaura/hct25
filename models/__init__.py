import copy
import timm
import torch


def build_model(model_name: str, num_classes: int, device: torch.device):
    """Create a model and its EMA copy."""
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    ema_model = copy.deepcopy(model)
    return model, ema_model

