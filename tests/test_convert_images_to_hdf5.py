import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, Mock

import h5py
import numpy as np
import pandas as pd
from PIL import Image


class UnitTest(TestCase):
    def create_blank_image(self, output_path: str, size: int):
        image = np.random.randint(low=0, high=255, size=(size, size, 3))
        image = Image.fromarray(np.uint8(image))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fp:
            print("Saved image", output_path)
            image.save(fp)
        return image

    def test_format_key(self):
        from src.core.h5py_tools import format_key

        self.assertEqual("cat_1|image1.png", format_key("cat_1/image1.png"))
        self.assertEqual("cat_1|image1.png", format_key("cat_1\\image1.png"))
        self.assertEqual(
            "some_folder|cat_1|image1.png", format_key("some_folder/cat_1\\image1.png")
        )

    def test_prepare_h5py_for_dataset(self):
        from src.core.h5py_tools import format_key, prepare_h5py_for_dataset
        from src.core.image_processing import create_crop_sequence

        output_folder = tempfile.mkdtemp()
        images_folder = tempfile.mkdtemp()
        try:
            size = 256
            config = Mock()
            config.augmentations.image_size = size
            crop_sequence = create_crop_sequence(config.augmentations, 42)
            images_df = pd.DataFrame(
                [
                    ["cat_1/image1.png", "cat_1"],
                    ["cat_1/image2.png", "cat_1"],
                    ["cat_2/image1.png", "cat_2"],
                ],
                columns=["path", "class"],
            )
            images_data = {
                path: self.create_blank_image(os.path.join(images_folder, path), size)
                for path in images_df["path"]
            }
            output_path = os.path.join(output_folder, "test.h5py")
            print("Output path", output_path)
            prepare_h5py_for_dataset(
                output_path, images_folder, images_df, crop_sequence
            )
            with h5py.File(output_path, "r") as file:
                for index, row in images_df.iterrows():
                    path = row["path"]
                    self.assertIn(format_key(path), file.keys())
                    self.assertEqual(
                        file[format_key(path)].attrs["class"],
                        row["class"],
                    )

                    data = np.array(images_data[path]).flatten().tolist()
                    actual_data = np.array(file[format_key(path)][:]).flatten().tolist()
                    self.assertEqual(data, actual_data)

        finally:
            shutil.rmtree(output_folder)
            shutil.rmtree(images_folder)
