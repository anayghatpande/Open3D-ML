import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath

from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from .utils import DataProcessing
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)

# Expect point clouds to be in npy format with train, val and test files in separate folders.
# Expected format of npy files : ['x', 'y', 'z', 'class', 'feat_1', 'feat_2', ........,'feat_n'].
# For test files, format should be : ['x', 'y', 'z', 'feat_1', 'feat_2', ........,'feat_n'].


class HOPE_dataset(BaseDataset):
    #Used for Custom dataset testing with HOPE dataset
    def __int__(self,
                dataset_path,
                name='HOPE',
                cache_dir='./logs/cache',
                use_cache=False,
                test_result_folder='./test',
                ignored_label_inds=[0],
                test_split=['00'],
                training_split=[
                    '01', '02', '03', '04', '05', '06', '07', '09'
                ],
                validation_split=['08'],
                all_split=[
                    '00', '01', '02', '03', '04', '05', '06', '07', '08', '09'
                ],
                **kwargs):
        """Initialize the dataset by passing the dataset and other details.
        Args:
            dataset_path (str): The path to the dataset to use.
            name (str): The name of the dataset (HOPE in this case).
            cache_dir (str): The directory where the cache is stored.
            num_points: The maximum number of points to use when splitting the dataset.
            use_cache (bool): Indicates if the dataset should be cached.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         test_split=test_split,
                         training_split=training_split,
                         validation_split=validation_split,
                         all_split=all_split,
                         **kwargs)
        cfg = self.cfg


        #self.num_classes = 28 #debug
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

        data_config = join(dirname(abspath(__file__)), '_resources/',
                           'hope_dataset_config.yml')
        DATA = yaml.safe_load(open(data_config, 'r'))
        #remap_dict = DATA["learning_map_inv"]

        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_dir = str(Path(cfg.dataset_path) / cfg.train_dir)
        #self.val_dir = str(Path(cfg.dataset_path) / cfg.val_dir)
        #self.test_dir = str(Path(cfg.dataset_path) / cfg.test_dir)

        self.train_files = [f for f in glob.glob(self.train_dir + "/*.npy")]
        #self.val_files = [f for f in glob.glob(self.val_dir + "/*.npy")]
        #self.test_files = [f for f in glob.glob(self.test_dir + "/*.npy")]


    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'obj_000001',
            1: 'obj_000002',
            2: 'obj_000003',
            3: 'obj_000004',
            4: 'obj_000005',
            5: 'obj_000006',
            6: 'obj_000007',
            7: 'obj_000008',
            8: 'obj_000009',
            9: 'obj_000010',
            10: 'obj_000011',
            11: 'obj_000012',
            12: 'obj_000013',
            13: 'obj_000014',
            14: 'obj_000015',
            15: 'obj_000016',
            16: 'obj_000017',
            17: 'obj_000018',
            18: 'obj_000019',
            19: 'obj_000020',
            20: 'obj_000021',
            21: 'obj_000022',
            22: 'obj_000023',
            23: 'obj_000024',
            24: 'obj_000025',
            25: 'obj_000026',
            26: 'obj_000027',
            27: 'obj_000028'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return HOPESplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'test', 'validation', or
             'all'.
        """
        if split in ['test', 'testing']:
            self.rng.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            self.rng.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            self.rng.shuffle(self.train_files)
            return self.train_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
            return files
        else:
            raise ValueError("Invalid split {}".format(split))


    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(self.label_to_names[pred])

        store_path = join(path, name + '.npy')
        np.save(store_path, pred)


class HOPESplit(BaseDatasetSplit):
    """This class is used to create a custom dataset split.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
        'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        data = np.load(pc_path)
        points = np.array(data[:, :3], dtype=np.float32)

        if (self.split != 'test'):
            labels = np.array(data[:, 3], dtype=np.int32)
            feat = data[:, 4:] if data.shape[1] > 4 else None
        else:
            feat = np.array(data[:, 3:],
                            dtype=np.float32) if data.shape[1] > 3 else None
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.npy', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}

        return attr



DATASET._register_module(HOPE_dataset)










