from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import torch

from src.core.metrics import TripletLoss
from src.core.model import create_score_model, create_train_model


class Test(TestCase):
    def test_TripletLoss(self):
        criterion = torch.nn.MSELoss()
        loss_function = TripletLoss(1.0, criterion)
        positive = torch.ones(64)
        anchor = torch.ones(64)
        negative = torch.zeros(64)
        loss = loss_function(positive, anchor, negative).item()
        estimated_loss = (
            torch.sum((positive - anchor) ** 2 - (negative - anchor) ** 2).item() / 64
            + 1.0
        )
        self.assertEqual(estimated_loss, loss)

        positive = torch.ones(64) + 2
        anchor = torch.ones(64)
        negative = torch.ones(64)
        loss = loss_function(positive, anchor, negative).item()
        estimated_loss = (
            torch.sum((positive - anchor) ** 2 - (negative - anchor) ** 2).item() / 64
            + 1.0
        )
        self.assertEqual(estimated_loss, loss)
        self.assertAlmostEqual(4 + 1, loss)

    def test_create_train_model(self):
        pretrained_name = "microsoft/resnet-18"
        output_vector_size = 64
        loss_function = TripletLoss(1.0, torch.nn.MSELoss())
        model = create_train_model(pretrained_name, output_vector_size, loss_function)
        positive_image = torch.ones((1, 3, 224, 224))
        anchor_image = torch.ones((1, 3, 224, 224))
        negative_image = torch.zeros((1, 3, 224, 224))

        positive_vector = model.single_forward(positive_image)
        anchor_vector = model.single_forward(anchor_image)
        negative_vector = model.single_forward(negative_image)

        loss = loss_function(positive_vector, anchor_vector, negative_vector).item()
        self.assertLess(loss, 1.0)
        loss_2, (positive_vector_2, anchor_vector_2, negative_vector_2) = model(
            positive_image, anchor_image, negative_image
        )
        self.assertAlmostEqual(loss, loss_2.item())
        self.assertListEqual(
            positive_vector.detach().numpy().flatten().tolist(),
            positive_vector_2.detach().numpy().flatten().tolist(),
        )
        self.assertListEqual(
            anchor_vector.detach().numpy().flatten().tolist(),
            anchor_vector_2.detach().numpy().flatten().tolist(),
        )
        self.assertListEqual(
            negative_vector.detach().numpy().flatten().tolist(),
            negative_vector_2.detach().numpy().flatten().tolist(),
        )

    def test_create_score_model(self):
        pretrained_name = "microsoft/resnet-18"
        output_vector_size = 64
        loss_function = TripletLoss(1.0, torch.nn.MSELoss())
        model = create_score_model(pretrained_name, output_vector_size, None)
        positive_image = torch.ones((1, 3, 224, 224))
        anchor_image = torch.ones((1, 3, 224, 224))
        negative_image = torch.zeros((1, 3, 224, 224))
        positive_vector = model(positive_image)
        anchor_vector = model(anchor_image)
        negative_vector = model(negative_image)
        loss = loss_function(positive_vector, anchor_vector, negative_vector).item()
        self.assertLess(loss, 1.0)
        self.assertEqual(positive_vector.shape[1], output_vector_size)
