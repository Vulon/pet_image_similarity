import torch
import numpy as np

def cos(a: torch.Tensor, b: torch.Tensor):
    loss = torch.nn.CosineSimilarity()(a, b)
    loss = torch.mean(torch.abs(loss))
    return loss


def create_compute_metrics_function():


    def compute_metrics(evalPrediction):
        positive_vector, anchor_vector, negative_vector = evalPrediction.predictions

        positive_mse = (positive_vector - anchor_vector)** 2
        negative_mse = (negative_vector - anchor_vector)** 2
        mask = negative_mse > positive_mse
        return {
            "accuracy": np.mean(mask), "positive mse" : np.mean(positive_mse), "negative_mse": np.mean(negative_mse),
            "positive cos" : cos(torch.tensor(positive_mse).float(), torch.tensor(anchor_vector).float()),
            "negative cos" : cos(torch.tensor(anchor_vector).float(), torch.tensor(negative_vector).float())
        }

    return compute_metrics