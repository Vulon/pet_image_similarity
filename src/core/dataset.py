from torch.utils.data import Dataset
import os
from collections import Counter
import random


class TripletLossDataset(Dataset):

    def __init__(self, images_folder_path: str, image_paths: list[str]):
        """

        :param images_folder_path:
        :param image_paths: list with relative image paths: should be class_folder/image_name
        """
        self.images_folder_path = images_folder_path

        self.classes = [os.path.dirname(path) for path in image_paths]
        self.single_classes = [ item[0] for item in Counter(self.classes).most_common() if item[1] < 2 ]
        self.image_paths = [os.path.basename(item) for item in image_paths]
        self.unique_classes = set(self.classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_class = self.classes[index]
        anchor_image_path = self.image_paths[index]

        if anchor_class in self.single_classes:
            # generate positive image with augmentations
            pass
        else:
            # sample positive image
            positive_index = random.sample([i for i, item in enumerate(self.classes) if item == anchor_class], 1)
            self.image_paths[positive_index]


