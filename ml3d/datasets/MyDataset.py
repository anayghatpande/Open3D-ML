import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
from open3d._ml3d.utils import make_dir

from .base_dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, name="MyDataset"):
        super().__init__(name=name)
        # read file lists.

    def get_split(self, split):

        return MyDatasetSplit(self, split=split)

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False
        # checks whether attr['name'] is already tested.

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(self.label_to_names[pred])

        store_path = join(path, name + '.npy')
        np.save(store_path, pred)
        # save results['predict_labels'] to file.


class MyDatasetSplit():
    def __init__(self, dataset, split='train'):
        self.split = split
        self.path_list = []
        # collect list of files relevant to split.

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path = self.path_list[idx]
        points, features, labels = read_pc(path)
        return {'point': points, 'feat': features, 'label': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1]
        return {'name': name, 'path': path, 'split': self.split}


DATASET._register_module(MyDataset)
