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
            image.save(fp)
        return image

    def test_prepare_h5py_for_dataset(self):
        from src.core.image_processing import create_crop_sequence
        from src.stages.convert_images_to_hdf5 import prepare_h5py_for_dataset

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
            prepare_h5py_for_dataset(
                output_path, images_folder, images_df, crop_sequence
            )
            with h5py.File(output_path, "r") as file:
                for path in images_data.keys():
                    self.assertIn(path, file.keys())
                    self.assertEqual(
                        file[path].attrs["class"],
                        images_df.loc[images_df["path"] == path, "class"].item(),
                    )

                    data = np.array(images_data[path]).flatten().tolist()
                    actual_data = np.array(file[path][:]).flatten().tolist()
                    self.assertEqual(data, actual_data)

        finally:
            shutil.rmtree(output_folder)
            shutil.rmtree(images_folder)
