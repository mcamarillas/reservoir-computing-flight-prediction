import torch 

def mape_loss(y_pred, y_true):
    epsilon = 1e-8
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))