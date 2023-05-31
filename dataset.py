from glob import glob
import cv2
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, source_contrast, contrast_list, transform):
        self.transform = transform
        self.data_dir = data_dir
        self.contrast_list = contrast_list
        self.data_path = {}
        self.source_contrast = source_contrast
        for contrast in self.contrast_list:
            self.data_path[contrast] = sorted(glob(f'{data_dir}/{contrast}/*'))

    def __len__(self):
        return len(self.data_path[self.contrast_list[0]])

    def __getitem__(self, idx):
        data = {}
        data['source'] = (
            self.transform(cv2.imread(self.data_path[self.source_contrast][idx])),
            [id for id in range(len(self.contrast_list)) if self.contrast_list[id] == self.source_contrast][0],
            self.data_path[self.source_contrast][idx]
        )
        data['target'] = {}
        for contrast in self.contrast_list:
            # if contrast != self.source_contrast:
            data['target'][contrast] = ((
                self.transform(cv2.imread(self.data_path[contrast][idx])),
                [id for id in range(len(self.contrast_list)) if self.contrast_list[id] == contrast][0],
                self.data_path[contrast][idx]
            ))
        return data
