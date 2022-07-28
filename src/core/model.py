import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    ResNetForImageClassification,
    SwinForImageClassification,
)


class SwinModel(nn.Module):
    def __init__(
        self, output_vector_size: int, criterion: nn.Module, pretrained_name: str
    ):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(pretrained_name)
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
    def __init__(
        self, output_vector_size: int, criterion: nn.Module, pretrained_name: str
    ):
        super().__init__()
        model = ResNetForImageClassification.from_pretrained(
            pretrained_name,
        )
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
    def __init__(self, output_vector_size: int, pretrained_name: str):
        super().__init__()
        model = ResNetForImageClassification.from_pretrained(pretrained_name)
        self.model = model.resnet
        self.decoder = nn.Linear(512, output_vector_size)

    def forward(self, image) -> torch.Tensor:
        output = self.model(image).pooler_output.squeeze(2).squeeze(2)
        output = self.decoder(output)
        return output


def create_train_model(
    pretrained_name: str, output_vector_size: int, loss_function: nn.Module
) -> nn.Module:
    table = {
        "microsoft/resnet-18": ResNet,
        "microsoft/swin-tiny-patch4-window7-224": SwinModel,
    }
    model = table[pretrained_name](
        output_vector_size=output_vector_size,
        criterion=loss_function,
        pretrained_name=pretrained_name,
    )
    return model


def create_score_model(
    pretrained_name: str, output_vector_size: int, saved_model_filepath: str
) -> nn.Module:
    table = {
        "microsoft/resnet-18": ResNetScore,
    }
    model = table[pretrained_name](
        output_vector_size=output_vector_size,
        pretrained_name=pretrained_name,
    )
    if saved_model_filepath:
        model.load_state_dict(torch.load(saved_model_filepath))
        model.eval()
    return model
