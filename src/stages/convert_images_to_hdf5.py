import h5py
import numpy as np
import pandas as pd
from imgaug.augmenters import Augmenter
import sys, os
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from src.config import get_config
from src.core.image_processing import create_crop_sequence, read_image


def prepare_h5py_for_dataset(output_filepath: str, images_folder_path: str, image_paths: pd.DataFrame, preprocessing_pipeline: Augmenter):
    with h5py.File(output_filepath, 'w') as file:
        for index, row in image_paths.iterrows():
            path = os.path.join(images_folder_path, row['path'] )
            image = read_image(path)
            processed_image = preprocessing_pipeline(images=[image])[0]
            processed_image = np.uint8(processed_image)
            file[row['path']] = processed_image
            file[row['path']].attrs["class"] = row['class']

if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    params = dvc.api.params_show()
    config = get_config(params)
    crop_sequence = create_crop_sequence(config.augmentations)

    prepare_h5py_for_dataset(
        os.path.join(project_root, config.data.train_h5_file),
        config.data.local_images_folder,
        pd.read_csv(os.path.join(project_root, config.data.train_classes_file)),
        crop_sequence
    )
    prepare_h5py_for_dataset(
        os.path.join(project_root, config.data.val_h5_file),
        config.data.local_images_folder,
        pd.read_csv(os.path.join(project_root, config.data.val_classes_file)),
        crop_sequence
    )
    prepare_h5py_for_dataset(
        os.path.join(project_root, config.data.test_h5_file),
        config.data.local_images_folder,
        pd.read_csv(os.path.join(project_root, config.data.test_classes_file)),
        crop_sequence
    )
