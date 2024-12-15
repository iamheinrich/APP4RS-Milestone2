# partial functions
from hashlib import md5
from typing import List, Literal, Optional, Callable

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


# additional imports
import pandas as pd
import numpy as np
import lmdb
import rasterio
from safetensors.numpy import load as safetensor_load
import math


def _hash(data):
    return md5(str(data).encode()).hexdigest()


BEN_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]
BEN_CLASSES.sort()
assert len(BEN_CLASSES) == 19, f"Expected 19 classes, got {len(BEN_CLASSES)}"

# Function to convert labels to one-hot encoded tensor. Necessary for multi-label classification.
def labels_to_onehot(labels, num_classes=19):
    onehot = torch.zeros(num_classes)
    for label in labels:
        idx = BEN_CLASSES.index(label)
        onehot[idx] = 1
    return onehot


BEN_BANDS = ["B01", "B02", "B03", "B04", "B05",
             "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]


class BENIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.lmdb_path = lmdb_path                                  #TODO this is the path to .lmdb not the .mdb inside it!!!
        self.bandorder = bandorder
        self.transform = transform

        self.metadata = pd.read_parquet(metadata_parquet_path)
        if split:
            self.metadata = self.metadata[self.metadata['split'] == split]

        # LMDB env will be initialized in worker processes to avoid parallel access issues
        self.env = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        # Open LMDB in current worker process if it wasn't opened through previous getitem call
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        # Get item's metadata
        metadata_row = self.metadata.iloc[idx]
        item_patch_name = metadata_row['patch_id']
        item_labels = metadata_row['labels']

        # Find item safetensor in lmdb through metadata
        with self.env.begin() as txn:
            tensor_bytes = txn.get(item_patch_name.encode())

        # Safetensor bytes to dict
        band_dict = safetensor_load(tensor_bytes)

        # Cat instead of stack selected bands
        resized_bands = []
        for band_name in self.bandorder:
            resized_bands.append(resize_band(band_dict[band_name]))

        reconstructed_patch = torch.cat(resized_bands)
        
        if self.transform:
            reconstructed_patch = self.transform(reconstructed_patch)

        # Convert class labels to one-hot encoded tensor
        index_labels = labels_to_onehot(item_labels)

        return reconstructed_patch, index_labels

def resize_band(uint16band): 
    band_tensor_unsqueezed = torch.tensor(uint16band, dtype=torch.float32).unsqueeze(0)
    band_tensor_resized = torch.nn.functional.interpolate(band_tensor_unsqueezed, size=(120, 120), mode='bilinear', align_corners=False).squeeze(0)
    return band_tensor_resized

class BENIndexableTifDataset(Dataset):
    def __init__(self, base_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using tif files.

        :param base_path: path to the source BigEarthNet dataset (root of the tar file)
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.base_path = base_path                          #TODO this leads to parent dir of BigEarthNet-Lithuania-Summer-S2 dir
        self.bandorder = bandorder
        self.transform = transform

        slash = "" if (base_path[-1] == "/") else "/"
        self.metadata = pd.read_parquet(base_path + slash + "lithuania_summer.parquet")
        if split:
            self.metadata = self.metadata[self.metadata['split'] == split]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        # Get item's metadata
        metadata_row = self.metadata.iloc[idx]
        item_patch_name = metadata_row['patch_id']
        item_labels = metadata_row['labels']

        path_to_patch = f"{self.base_path}/BigEarthNet-Lithuania-Summer-S2/{item_patch_name[:-6]}/{item_patch_name}"

        resized_bands = []

        # Select bands in bandorder as in method argument
        for band_name in self.bandorder:
            tif_path = f"{path_to_patch}/{item_patch_name}_{band_name}.tif"
            with rasterio.open(tif_path) as src:
                resized_bands.append(resize_band(src.read()))                    #TODO not src.read(1) because of shape to enable stacking

        # Cat instead of stacking, "useless" channel dim
        patch = torch.cat(resized_bands)
        
        if self.transform:
            patch = self.transform(patch)

        # Convert class labels to one-hot encoded tensor
        index_labels = labels_to_onehot(item_labels)

        return patch, index_labels


class BENIterableLMDBDataset(IterableDataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None,
                 with_keys=False):
        """
        IterableDataset for the BigEarthNet dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: order of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """

        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.bandorder = bandorder
        self.split = split
        self.transform = transform
        self.with_keys = with_keys

        # Load metadata and filter by split if provided to reduce memory usage
        metadata = pd.read_parquet(self.metadata_parquet_path)
        if self.split is not None:
            metadata = metadata[metadata['split'] == self.split]

        self.metadata = metadata
        self.labels_dict = dict(zip(metadata['patch_id'], metadata['labels']))  # O(1) lookup
        self.keys = metadata['patch_id'].tolist()

        # Check if all bands from bandorder are in BEN_BANDS
        for band in self.bandorder:
            assert band in BEN_BANDS, f"Band {band} not found in BEN_BANDS"
            
        # LMDB env will be initialized in worker processes to avoid parallel access issues
        self.env = None

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        """
        Iterate over the dataset.

        :return: an iterator over the dataset, e.g. via `yield` where each item is a (patch, label) tuple where patch is
            a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """

        # Create a connection to the lmdb file
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        # Get worker info
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Split workload so that each worker can process a different subset of the data.
            # We can achieve this by determining the number of samples per worker.
            per_worker = int(math.ceil(self.__len__() /float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.__len__())
            iter_keys = self.keys[iter_start:iter_end]
        else:
            iter_keys = self.keys

        with self.env.begin(write=False) as txn:
            for key in iter_keys:
                # Get the dict of bands for key (patch_id)
                st = txn.get(key.encode())
                assert st is not None, f"Key {key} not found in LMDB"
                band_dict = safetensor_load(st)

                # Check if the keys of the band dict are the same as the keys in BEN_BANDS
                band_dict_keys = list(band_dict.keys())
                assert set(band_dict_keys) == set(BEN_BANDS), f"Expected band dict keys to be {BEN_BANDS}, got {band_dict_keys}"

                # Images/Arrays for each band are stored in a list.
                # The first image corresponds to the first band in bandorder, the second to the second band, etc.
                # Use np.stack to ensure that it fails if the dimensions of the arrays are not the same.
                patch = torch.cat([resize_band(band_dict[band])for band in self.bandorder])


                # Check if the dimensions of band arrays are 3 (C, H, W)
                assert len(patch.shape) == 3, "Expected 3D array for band arrays"

                # Apply the transform to torch tensor of band arrays if transform is provided
                if self.transform:
                    patch = self.transform(patch)

                # Convert labels to tensor of shape (N,) assuming that the label corresponds to the index of the class in BEN_CLASSES
                # Labels is a list of strings, where each string corresponds to the class of the patch.
                labels = self.labels_dict[key]
                # Check if numpy array contains strings as labels
                assert all(isinstance(label, str) for label in labels), f"Expected labels to be a list of strings, got {labels.dtype}"

                # Convert class labels to one-hot encoded tensor
                index_labels = labels_to_onehot(labels)

                yield patch, labels


class BENDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            bandorder: List,
            ds_type: Literal['iterable_lmdb', 'indexable_tif', 'indexable_lmdb'],
            base_path: Optional[str] = None,
            lmdb_path: Optional[str] = None,
            metadata_parquet_path: Optional[str] = None,
            g: Optional[torch.Generator] = None,
            worker_init_fn: Optional[Callable] = None,
    ):
        """
        DataModule for the BigEarthNet dataset.

        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers for the dataloaders
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param ds_type: type of dataset to use, one of 'iterable_lmdb', 'indexable_tif', 'indexable_lmdb'
        :param base_path: path to the source BigEarthNet dataset (root of the tar file), for tif dataset
        :param lmdb_path: path to the converted lmdb file, for lmdb dataset
        :param metadata_parquet_path: path to the metadata parquet file, for lmdb dataset
        :param g: torch generator for reproducibility
        :param worker_init_fn: function to initialize worker
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bandorder = bandorder
        self.ds_type = ds_type
        self.base_path = base_path
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.g = g
        self.worker_init_fn = worker_init_fn

    def setup(self, stage=None):
        # Based on ds_type, select the appropriate dataset class and keyword args
        if self.ds_type == 'iterable_lmdb':
            DS = BENIterableLMDBDataset
            ds_kwargs = {
                'lmdb_path': self.lmdb_path,
                'metadata_parquet_path': self.metadata_parquet_path,
                'bandorder': self.bandorder
            }
        elif self.ds_type == 'indexable_tif':
            DS = BENIndexableTifDataset
            ds_kwargs = {
                'base_path': self.base_path,
                'bandorder': self.bandorder
            }
        elif self.ds_type == 'indexable_lmdb':
            DS = BENIndexableLMDBDataset
            ds_kwargs = {
                'lmdb_path': self.lmdb_path,
                'metadata_parquet_path': self.metadata_parquet_path,
                'bandorder': self.bandorder
            }
        else:
            raise ValueError(f"Unknown ds_type: {self.ds_type}")
        
        # Create dataset objects for train, validation and test splits
        self.train_dataset = DS(split='train', **ds_kwargs)
        self.val_dataset = DS(split='validation', **ds_kwargs)
        self.test_dataset = DS(split='test', **ds_kwargs)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.g,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.g,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.g,
            worker_init_fn=self.worker_init_fn,
        )


############################################ DON'T CHANGE CODE BELOW HERE ############################################


def main(
        lmdb_path: str,
        metadata_parquet_path: str,
        tif_base_path: str,
        bandorder: List,
        sample_indices: List[int],
        num_batches: int,
        seed: int,
        timing_samples: int,
):
    """
    Test the BigEarthNet dataset classes.

    :param lmdb_path: path to the converted lmdb file
    :param metadata_parquet_path: path to the metadata parquet file
    :param tif_base_path: path to the source BigEarthNet dataset (root of the tar file)
    :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
    :param sample_indices: indices of samples to check for correctness
    :param num_batches: number of batches to check in the dataloaders for correctness
    :param seed: seed for the dataloaders for reproducibility
    :param timing_samples: number of samples to check during timing
    :return: None
    """
    import time

    # check values of sample_indices
    for split in ['train', 'validation', 'test', None]:
        print(f"\nSplit: {split}")
        for DS in [BENIndexableLMDBDataset, BENIndexableTifDataset, BENIterableLMDBDataset]:
            paths = {
                "base_path": tif_base_path,
            } if DS == BENIndexableTifDataset else {
                "lmdb_path": lmdb_path,
                "metadata_parquet_path": metadata_parquet_path,
            }
            # create dataset
            ds = DS(
                bandorder=bandorder,
                split=split,
                **paths
            )
            # check values of sample_indices, collect hashes
            total_str = ""
            if DS == BENIterableLMDBDataset:
                for i, (x, y) in enumerate(ds):
                    total_str += _hash(x) + _hash(y)
                    if i >= len(sample_indices):
                        break
            else:
                for i in sample_indices:
                    x, y = ds[i]
                    total_str += _hash(x) + _hash(y)

            # check timing
            t0 = time.time()
            for i, _ in enumerate(iter(ds)):
                if i >= timing_samples:
                    break
            ds_type = "IterableLMDB " if DS == BENIterableLMDBDataset \
                else "IndexableTif " if DS == BENIndexableTifDataset \
                else "IndexableLMDB"
            print(f"{split}-{ds_type}: {_hash(total_str)} @ {time.time() - t0:.2f}s")

    print()
    for ds_type in ['indexable_lmdb', 'indexable_tif', 'iterable_lmdb']:
        # seed the dataloaders for reproducibility
        torch.manual_seed(seed)
        dm = BENDataModule(
            batch_size=1,
            num_workers=0,
            bandorder=bandorder,
            ds_type=ds_type,
            lmdb_path=lmdb_path,
            metadata_parquet_path=metadata_parquet_path,
            base_path=tif_base_path
        )
        dm.setup()
        # again, collect hashes
        total_str = ""
        for i in range(num_batches):
            for x, y in dm.train_dataloader():
                total_str += _hash(x) + _hash(y)
                break
            for x, y in dm.val_dataloader():
                total_str += _hash(x) + _hash(y)
                break
            for x, y in dm.test_dataloader():
                total_str += _hash(x) + _hash(y)
                break
        print(f"datamodule-{ds_type:<14}: {_hash(total_str)}")
