from glob import glob

import numpy as np
import cv2

import torch 
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image

class BratDataset(Dataset):
    def __init__(self, data_dir, transform=None, source_format='flair'):
        self.transform = transform
        self.data_dir = data_dir
        self.format_list = ['flair', 't1', 't1ce', 't2']
        self.data_path = {}
        self.source_format = source_format
        for format in self.format_list:
            self.data_path[format] = sorted(glob(f'{data_dir}/{format}/*'))

    def __len__(self):
        return len(self.data_path[self.format_list[0]])

    def __getitem__(self, idx):
        data = {}
        data['source'] = (
            self.transform(cv2.imread(self.data_path[self.source_format][idx])),
            [id for id in range(len(self.format_list)) if self.format_list[id] == self.source_format][0],
            self.data_path[self.source_format][idx]
        )
        data['target'] = []
        for format in self.format_list:
            if format != self.source_format:
                data['target'].append((
                    self.transform(cv2.imread(self.data_path[format][idx])),
                    [id for id in range(len(self.format_list)) if self.format_list[id] == format][0],
                    self.data_path[format][idx]
                ))
        return data
