import os
import sys

import pandas as pd

from src.core.h5py_tools import prepare_h5py_for_dataset

if __name__ == "__main__":
    sys.path.append(os.environ["DVC_ROOT"])
    from src.config import get_config_from_dvc
    from src.core.image_processing import create_crop_sequence

    project_root = os.environ["DVC_ROOT"]
    config = get_config_from_dvc()
    crop_sequence = create_crop_sequence(config.augmentations, config.random_seed)

    prepare_h5py_for_dataset(
        os.path.join(project_root, config.data.train_h5_file),
        config.data.local_images_folder,
        pd.read_csv(os.path.join(project_root, config.data.train_classes_file)),
        crop_sequence,
    )
    prepare_h5py_for_dataset(
        os.path.join(project_root, config.data.val_h5_file),
        config.data.local_images_folder,
        pd.read_csv(os.path.join(project_root, config.data.val_classes_file)),
        crop_sequence,
    )
    prepare_h5py_for_dataset(
        os.path.join(project_root, config.data.test_h5_file),
        config.data.local_images_folder,
        pd.read_csv(os.path.join(project_root, config.data.test_classes_file)),
        crop_sequence,
    )
