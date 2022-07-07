import h5py
import numpy as np
from torch.utils.data import Dataset
import os
from collections import Counter
import random
import pandas as pd
from imgaug.augmenters import Augmenter
import random


class ImageDataset(Dataset):

    def __init__(self, dataset_path: str, augmenter: Augmenter, use_augmenter=False, ):
        """
        :param images_folder_path:
        :param image_paths: list with relative image paths: should be class_folder/image_name
        """
        self.augmenter = augmenter
        self.use_augmenter = use_augmenter
        self.classes = dict()
        self.paths_by_classes = dict()
        self.paths = []
        self.file = h5py.File(dataset_path, "r")
        image_paths = self.file.keys()
        for path in image_paths:
            image_class = self.file[path].attrs["class"]
            self.classes[path] = image_class
            paths_list = self.paths_by_classes.get(image_class, [])
            paths_list.append(path)
            self.paths_by_classes[image_class] = paths_list
            self.paths.append(path)

        self.unique_classes = set(self.classes.values())
        self.single_classes = [item[0] for item in Counter(self.classes.values()).most_common() if item[1] < 2]

    def __len__(self):
        return len(self.paths)

    def augment(self, image):
        return self.augmenter(images=[image])[0]

    def __getitem__(self, index):
        anchor_path = self.paths[index]
        anchor_class = self.classes[anchor_path]
        anchor_image = self.file[anchor_path][:]
        negative_class = random.choice( [cls for cls in self.unique_classes if cls != anchor_class] )
        negative_path = random.choice(self.paths_by_classes[negative_class])
        negative_image = self.file[negative_path][:]

        if anchor_class in self.single_classes:
            positive_image = self.augment(anchor_image)
        else:
            positive_path = random.choice([item for item in self.paths_by_classes[anchor_class] if item != anchor_path])
            positive_image = self.file[positive_path][:]
            if self.use_augmenter:
                positive_image = self.augment(positive_image)
        if self.use_augmenter:
            anchor_image = self.augment(anchor_image)
            negative_image = self.augment(negative_image)
        return { "positive": positive_image, "anchor": anchor_image, "negative": negative_image }

    def __del__(self):
        self.file.close()
