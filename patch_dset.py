"""Torch Dataset that yields 'patches' from seismic gathers."""
import random
import pickle
import itertools
import operator
import math
import os
import pathlib

import numpy
import torch
from torch.utils.data import Dataset, DataLoader

# pytorch lightning
import pytorch_lightning as pl

# https://docs.python.org/3/library/itertools.html
def iter_index(iterable, value, start=0):
    "Return indices where a value occurs in a sequence or iterable."
    # iter_index('AABCADEAF', 'A') --> 0 1 4 7
    try:
        seq_index = iterable.index
    except AttributeError:
        # Slow path for general iterables
        it = itertools.islice(iterable, start, None)
        i = start - 1
        try:
            while True:
                yield (i := i + operator.indexOf(it, value) + 1)
        except ValueError:
            pass
    else:
        # Fast path for sequences
        i = start - 1
        try:
            while True:
                yield (i := seq_index(value, i+1))
        except ValueError:
            pass

class PatchDataset(Dataset):
    """Dataset class to yield a patch from seismic gather.
    
    Attributes
    ----------
    

    Methods
    -------
    """
    
    def __init__(self, max_fold:int, num_samples:int, patch_size:int, mode:str='train', patch_index_mapping=None):
        """
        Parameters
        ----------
        mode : str, optinal
            Either 'train', 'test', or 'valid'. Used to determine which directory to pull data from. Default == 'train'.    

        Raises
        ------
        """
        self.patch_size = patch_size
        self.max_fold = max_fold  # must be less than or equal to data.shape[1]
        self.num_samples = num_samples  # must be less than or equal to data.shape[2]
        # npy files in the data directory
        self.ensemble_files = []
        file_dir = pathlib.Path.cwd().joinpath('data')
        for p in file_dir.iterdir():
            if p.suffix == '.npy':
                self.ensemble_files.append(p)
        
        if patch_index_mapping:
            self.patch_from_index = patch_index_mapping
        else:
            self.patch_from_index = self.create_index_mapping(
                mode=mode, 
                num_ensembles=len(self.ensemble_files), 
                max_fold=max_fold, 
                num_samples=num_samples, 
                patch_size=patch_size, 
                testing_fraction=0.2,  # could be dataset hyper-parameter 
                validation_fraction=0.1)  # could be dataset hyper-parameter
        

    def __len__(self):
        """The number of patches."""
        return len(self.patch_from_index)

    def __getitem__(self, idx):
        """Returns source and target data. 
        """
        # lazy loading; read the numpy file (single CMP ensemble) and return the patch
        p_idx = self.patch_from_index[idx]
        ensemble = numpy.load(self.ensemble_files[p_idx[0]])
        source_patch = ensemble[p_idx[1]:p_idx[1]+self.patch_size, p_idx[2]:p_idx[2]+self.patch_size].copy()  # patch is (fold, samples) in a form unfamiliar to geoscientists; use .transpose()
        sclr = numpy.max(numpy.abs(source_patch))
        if sclr != 0:
            source_patch = source_patch / sclr  # normalizes the amplitudes to range (-1,1)
        source_patch = torch.tensor(source_patch).reshape((1,self.patch_size, self.patch_size))  # NOTE: ASSUMES ONLY 1 CHANNEL!!!
        target_patch = source_patch.clone().detach()
        return source_patch, target_patch
    
    @staticmethod
    def create_index_mapping(mode:str, num_ensembles:int, max_fold:int, num_samples:int, patch_size:int, testing_fraction:float=0.2, validation_fraction:float=0.1):
        patch_tuples = [i for i in itertools.product([i for i in range(num_ensembles)], [j for j in range(max_fold - patch_size+1)], [k for k in range(num_samples - patch_size+1)])]
        # 0 for train, 1 for test, 2 for valid
        twos = [t for t in itertools.repeat(2, math.ceil(len(patch_tuples) * validation_fraction))]
        ones = [t for t in itertools.repeat(1, math.ceil(len(patch_tuples) * testing_fraction))]
        zeros = [t for t in itertools.repeat(0, len(patch_tuples) - len(ones) - len(twos))]
        training_splits = zeros + ones + twos  # concatenate lists
        random.shuffle(training_splits)
        # unique index value for each patch from the input data
        if mode == 'train':
            return {i:patch_tuples[j] for i,j in enumerate(iter_index(training_splits, 0))}
        elif mode == 'test':
            return {i:patch_tuples[j] for i,j in enumerate(iter_index(training_splits, 1))}
        elif mode == 'valid':
            return {i:patch_tuples[j] for i,j in enumerate(iter_index(training_splits, 2))}
        else:
            raise ValueError(f"mode {mode} is not defined.")
        
        

# https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
class PatchDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, max_fold:int, num_samples:int, patch_size:int):
        super().__init__()
        self.batch_size = batch_size
        self.max_fold = max_fold
        self.num_samples = num_samples
        self.patch_size = patch_size

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.training_dataset = PatchDataset(max_fold=self.max_fold, num_samples=self.num_samples, patch_size=self.patch_size, mode='train')
            self.validating_dataset = PatchDataset(max_fold=self.max_fold, num_samples=self.num_samples, patch_size=self.patch_size, mode='valid')

        if stage == "test" or stage is None:
            self.testing_dataset = PatchDataset(max_fold=self.max_fold, num_samples=self.num_samples, patch_size=self.patch_size, mode='test')

    ## For dataloaders, usually just wrap dataset defined in setup
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.validating_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)