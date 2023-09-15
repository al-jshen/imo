#!/usr/bin/env python
# coding: utf-8

import io, os, sys, time, random
import numpy as np
import pickle
import torch
from torch.utils.data import IterableDataset, DataLoader

from itertools import chain


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class BatchedFilesDataset(IterableDataset):
    def __init__(self, file_list, load_fct, shuffle=False, shuffle_instance=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.shuffle_instance = shuffle_instance
        self.load_fct = load_fct

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) - 1)
        batch_name = self.file_list[idx]
        data = self.load_fct(batch_name)
        data = list(zip(*data))
        if self.shuffle_instance:
            random.shuffle(data)
        for x in data:
            yield x

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))

    def __iter__(self):
        return self.get_stream()

    def __len__(self):
        return len(self.file_list)


from functools import partial


def load_batch(batch_name):
    # print("batch_name:",batch_name)
    with open(batch_name, "rb") as f:
        if torch.cuda.is_available():
            batch = pickle.load(f)
        else:
            batch = CPU_Unpickler(f).load()
    batch = [item.detach().to(device) for item in batch]
    return batch


def get_latent_data_loader(
    dir,
    which=None,
    batch_size=10000,
    shuffle=False,
    shuffle_instance=True,
    latent_tag=None,
):
    files = ["%s/%s" % (dir, item) for item in os.listdir(dir)]
    if latent_tag is not None:
        files = [item for item in files if latent_tag in item]
    NBATCH = len(files)
    train_batches = files[: int(0.85 * NBATCH)]
    valid_batches = files[int(0.85 * NBATCH) :]

    if which == "valid":
        files = valid_batches
    elif which == "train":
        files = train_batches

    load_fct = partial(load_batch)
    data = BatchedFilesDataset(
        files, load_fct, shuffle=shuffle, shuffle_instance=shuffle_instance
    )
    return DataLoader(data, batch_size=batch_size)
