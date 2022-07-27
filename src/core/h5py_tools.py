import os

import h5py
import numpy as np
import pandas as pd
from imgaug.augmenters import Augmenter


def format_key(path: str) -> str:
    return path.replace("\\", "|").replace("/", "|")


def prepare_h5py_for_dataset(
    output_filepath: str,
    images_folder_path: str,
    image_paths: pd.DataFrame,
    preprocessing_pipeline: Augmenter,
):
    from src.core.image_processing import read_image

    with h5py.File(output_filepath, "w") as file:
        for index, row in image_paths.iterrows():
            path = os.path.join(images_folder_path, row["path"])
            image = read_image(path)
            processed_image = preprocessing_pipeline(images=[image])[0]
            processed_image = np.uint8(processed_image)

            path_key = format_key(row["path"])
            file[path_key] = processed_image
            file[path_key].attrs["class"] = row["class"]
