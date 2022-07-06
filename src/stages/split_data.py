from collections import Counter
import os
import tqdm
import pandas as pd
import sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from src.config import get_config


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    params = dvc.api.params_show()
    config = get_config(params)
    df = pd.read_csv( os.path.join(project_root, config.data.classes_text_file) )
    classes = Counter(df['class']).most_common()
    total_count = df.shape[0]
    train_count = int(total_count * config.data.train_fracture)
    val_count = int(total_count * config.data.val_fracture)
    test_count = int(total_count * config.data.test_fracture)
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
    total_count = df.shape[0]
    train_df = df.loc[df['class'].isin(train_classes)]
    val_df = df.loc[df['class'].isin(val_classes)]
    test_df = df.loc[df['class'].isin(test_classes)]
    print("Desired portions:", config.data.train_fracture, config.data.val_fracture, config.data.test_fracture)
    print("Final portions:", train_df.shape[0] / total_count, val_df.shape[0] / total_count, test_df.shape[0] / total_count)
    train_df.to_csv(os.path.join(project_root, config.data.train_classes_file))
    val_df.to_csv(os.path.join(project_root, config.data.val_classes_file))
    test_df.to_csv(os.path.join(project_root, config.data.test_classes_file))

