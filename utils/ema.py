import torch

@torch.no_grad()
def update_ema_model(model, ema_model, decay=0.999):
    """
    Updates the EMA (Exponential Moving Average) model parameters.

    Args:
        model (torch.nn.Module): The current model.
        ema_model (torch.nn.Module): The model tracking EMA of weights.
        decay (float): The decay rate for EMA. Should be close to 1.0.
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
