import os
import cv2
import torch
import tqdm
import pandas as pd
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
import sys
import dvc.api
project_root = os.environ["DVC_ROOT"]
sys.path.append(project_root)
from src.config import get_config
from src.core.image_processing import create_train_sequence
from src.core.dataset import ImageDataset
from imgaug import augmenters as iaa
from src.core.model import ResNetScore
import numpy as np
from transformers import AutoFeatureExtractor
from src.core.metrics import cos




def create_table(h5py_file: h5py.File, feature_extractor, model):
    table = []
    for i, key in enumerate(h5py_file.keys()):
        image_class = h5py_file[key].attrs["class"]
        model_input = feature_extractor(images=h5py_file[key][:], return_tensors="pt")["pixel_values"]

        vector = model.forward(model_input).detach().numpy().flatten()
        table.append([key, image_class, torch.tensor(vector), 0])
    return table

def print_image(image_data, image_class, distance, ax_row):
    ax_row[0].imshow(image_data)
    ax_row[1].text(0.3, 0.7, image_class)
    ax_row[1].text(0.3, 0.3, distance)
    ax_row[1].set_axis_off()
    ax_row[0].set_axis_off()

def create_similarity_plot(table: list, loss_function, h5py_file: h5py.File):
    fig, axes = plt.subplots(nrows=len(table), ncols=2, figsize=( 8, len(table) * 4,  ))

    image_path, image_class, vector, distance = table.pop(0)
    print_image(h5py_file[image_path][:], image_class, distance, axes[0])

    for i in range(1, len(table) + 1):
        table = [ (item[0], item[1], item[2], loss_function(vector, item[2]).item()) for item in table]
        table = sorted(table, key=lambda x : x[3])
        image_path, image_class, vector, distance = table.pop(0)
        print_image(h5py_file[image_path][:], image_class, distance, axes[i])

    plt.axis('off')
    return fig

if __name__ == "__main__":
    params = dvc.api.params_show()
    config = get_config(params)
    sequence = create_train_sequence(config.augmentations)
    model = ResNetScore(config.model.output_vector_size)
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model.pretrained_model_name)

    if config.trainer.loss_function == "cos":
        loss_function = cos
    elif config.trainer.loss_function == "mse":
        loss_function = torch.nn.MSELoss()

    test_file = h5py.File(os.path.join(project_root, config.data.test_h5_file))
    test_file.close()
    plt.savefig(os.path.join(project_root, "data", "test_visualization.png"))
