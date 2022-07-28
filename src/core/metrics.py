import numpy as np
import torch
from torch import nn


class CosLoss(nn.Module):
    def _cos(self, a: torch.Tensor, b: torch.Tensor):
        loss = torch.nn.CosineSimilarity()(a, b)
        loss = torch.mean(torch.abs(loss))
        return loss

    def forward(self, first: torch.Tensor, second: torch.Tensor):
        return self._cos(first, second)


def create_compute_metrics_function():
    cos = CosLoss()

    def compute_metrics(evalPrediction):
        positive_vector, anchor_vector, negative_vector = evalPrediction.predictions

        positive_mse = (positive_vector - anchor_vector) ** 2
        negative_mse = (negative_vector - anchor_vector) ** 2
        mask = negative_mse > positive_mse
        return {
            "accuracy": np.mean(mask),
            "positive mse": np.mean(positive_mse),
            "negative_mse": np.mean(negative_mse),
            "positive cos": cos(
                torch.tensor(positive_mse).float(), torch.tensor(anchor_vector).float()
            ),
            "negative cos": cos(
                torch.tensor(anchor_vector).float(),
                torch.tensor(negative_vector).float(),
            ),
        }

    return compute_metrics


class TripletLoss(nn.Module):
    def __init__(self, alpha, loss_function):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.loss_function = loss_function

    def forward(self, positive_vector, anchor_vector, negative_vector):
        loss = (
            self.loss_function(positive_vector, anchor_vector)
            - self.loss_function(anchor_vector, negative_vector)
            + self.alpha
        )
        return torch.clip(loss, min=0)


def get_base_loss_function(base_loss_function_name: str) -> nn.Module:
    if base_loss_function_name == "cos":
        loss_function = CosLoss()
    elif base_loss_function_name == "mse":
        loss_function = torch.nn.MSELoss()

    return loss_function
