import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    ResNetForImageClassification,
    SwinForImageClassification,
)


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


class SwinModel(nn.Module):
    def __init__(self, output_vector_size: int, criterion: nn.Module):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        self.model = model.swin
        self.decoder = nn.Linear(768, output_vector_size)
        self.criterion = criterion

    def single_forward(self, image):
        output = self.model(image).pooler_output
        output = self.decoder(output)
        return output

    def forward(self, positive, anchor, negative, **kwargs):
        positive_vector = self.single_forward(positive)
        anchor_vector = self.single_forward(anchor)
        negative_vector = self.single_forward(negative)

        loss = self.criterion(positive_vector, anchor_vector, negative_vector)
        return (loss, anchor_vector)


class ResNet(nn.Module):
    def __init__(self, output_vector_size: int, criterion: nn.Module):
        super().__init__()
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
        self.model = model.resnet
        self.decoder = nn.Linear(512, output_vector_size)
        self.criterion = criterion

    def single_forward(self, image):
        output = self.model(image).pooler_output.squeeze(2).squeeze(2)
        output = self.decoder(output)
        return output

    def forward(self, positive, anchor, negative, **kwargs):
        positive_vector = self.single_forward(positive)
        anchor_vector = self.single_forward(anchor)
        negative_vector = self.single_forward(negative)

        loss = self.criterion(positive_vector, anchor_vector, negative_vector)
        return (loss, [positive_vector, anchor_vector, negative_vector])


class ResNetScore(nn.Module):
    def __init__(self, output_vector_size: int):
        super().__init__()
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
        self.model = model.resnet
        self.decoder = nn.Linear(512, output_vector_size)

    def forward(self, image) -> torch.Tensor:
        output = self.model(image).pooler_output.squeeze(2).squeeze(2)
        output = self.decoder(output)
        return output
