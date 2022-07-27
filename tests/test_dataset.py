import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import Mock

import h5py
import numpy as np
import torch
from imgaug import augmenters as iaa
from PIL import Image

from src.core.dataset import ImageDataset
from src.core.h5py_tools import format_key


class TestImageDataset(TestCase):
    def create_blank_image(self, size: int):
        image = np.random.randint(low=0, high=255, size=(size, size, 3))
        image = Image.fromarray(np.uint8(image))
        return image

    def create_h5py_file(self):
        self.temp_folder = tempfile.mkdtemp()
        self.temp_file_path = os.path.join(self.temp_folder, "temp_file.h5py")
        self.images = []
        with h5py.File(self.temp_file_path, "w") as file:
            paths = [
                "cat_1/image1.png",
                "dog_2/image_3.png",
                "cat_1/image2.png",
                "some_folder/cat_3/image4.png",
            ]

            self.image_paths = paths
            self.image_classes = [
                os.path.basename(os.path.dirname(item)) for item in paths
            ]
            for path in paths:
                image = self.create_blank_image(256)
                self.images.append(image)
                file[format_key(path)] = image
                file[format_key(path)].attrs["class"] = os.path.basename(
                    os.path.dirname(path)
                )

            print("File keys", file.keys())

    def setUp(self) -> None:
        self.create_h5py_file()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_folder)

    def test_choose_negative_path(self):
        augmenter = iaa.CenterCropToSquare()
        mock_feature_extractor = Mock(
            return_value={"pixel_values": torch.ones(1, 64, 64, 3)}
        )
        dataset = ImageDataset(
            self.temp_file_path,
            augmenter,
            mock_feature_extractor,
            use_augmenter=False,
        )

        negative_path = dataset.choose_negative_path("cat_1")
        self.assertNotEqual("cat_1", negative_path)
        negative_path = dataset.choose_negative_path("cat_2")
        self.assertNotEqual("cat_2", negative_path)

        positive_path = dataset.choose_positive_path("cat_1", "cat_1|image1.png")
        self.assertEqual(positive_path, "cat_1|image2.png")
        positive_path = dataset.choose_positive_path("cat_1", "cat_1|image2.png")
        self.assertEqual(positive_path, "cat_1|image1.png")

        tensor = dataset.apply_feature_extractor(self.images[0])
        self.assertEqual(type(tensor), torch.Tensor)
        mock_feature_extractor.assert_called_once()

        results = dataset[0]
        self.assertIn("positive", results.keys())
        self.assertIn("anchor", results.keys())
        self.assertIn("negative", results.keys())
        self.assertEqual(type(results["positive"]), torch.Tensor)
        self.assertEqual(type(results["anchor"]), torch.Tensor)
        self.assertEqual(type(results["negative"]), torch.Tensor)
