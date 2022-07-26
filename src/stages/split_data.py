import os
import sys
from collections import Counter

import dvc.api
import pandas as pd
import tqdm


def _split_classes(
    classes_list: list[str],
    train_fracture: float,
    val_fracture: float,
    test_fracture: float,
) -> tuple[list[str], list[str], list[str]]:
    classes = Counter(classes_list).most_common()
    total_count = len(classes_list)
    train_fracture = train_fracture / (train_fracture + val_fracture + test_fracture)
    val_fracture = val_fracture / (train_fracture + val_fracture + test_fracture)
    test_fracture = test_fracture / (train_fracture + val_fracture + test_fracture)
    train_count = int(total_count * train_fracture)
    val_count = int(total_count * val_fracture)
    test_count = int(total_count * test_fracture)
    train_classes, val_classes, test_classes = [], [], []
    for i, (class_name, class_size) in enumerate(classes):
        if class_size < test_count and i % 3 == 0:
            test_classes.append(class_name)
            test_count -= class_size
        elif class_size < val_count and i % 3 == 1:
            val_classes.append(class_name)
            val_count -= class_size
        else:
            train_classes.append(class_name)
            train_count -= class_size
    return train_classes, val_classes, test_classes


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    sys.path.append(project_root)
    from src.config import get_config_from_dvc

    config = get_config_from_dvc()

    df = pd.read_csv(os.path.join(project_root, config.data.classes_text_file))

    train_classes, val_classes, test_classes = _split_classes(
        df["class"],
        config.data.train_fracture,
        config.data.val_fracture,
        config.data.test_fracture,
    )
    train_df = df.loc[df["class"].isin(train_classes)]
    val_df = df.loc[df["class"].isin(val_classes)]
    test_df = df.loc[df["class"].isin(test_classes)]

    train_df.to_csv(
        os.path.join(project_root, config.data.train_classes_file), index=False
    )
    val_df.to_csv(os.path.join(project_root, config.data.val_classes_file), index=False)
    test_df.to_csv(
        os.path.join(project_root, config.data.test_classes_file), index=False
    )
