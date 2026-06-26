import os
import random
import numpy as np
from abc import ABCMeta, abstractmethod
import collections
from omegaconf import OmegaConf
import omegaconf
import logging

import torch
from torch.utils.data import DataLoader, Sampler, get_worker_info
import torch.distributed as dist
# from ..utils.tools import set_num_threads, set_seed
# from torch._six import string_classes
string_classes = str
from torch.utils.data._utils.collate import (default_collate_err_msg_format,
                                             np_str_obj_array_pattern)
from pytorch3d.structures import join_meshes_as_batch, Meshes

logger = logging.getLogger(__name__)

def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ['IPC_ENABLE'] = '1'
    for o in ['OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[o] = str(nt)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class LoopSampler(Sampler):
    def __init__(self, loop_size, total_size=None):
        self.loop_size = loop_size
        self.total_size = total_size - (total_size % loop_size)

    def __iter__(self):
        return (i % self.loop_size for i in range(self.total_size))

    def __len__(self):
        return self.total_size

def collate(batch):
    """Difference with PyTorch default_collate: it can stack of other objects.
        """
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
        # return torch.stack(batch, dim=0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, Meshes):
        return join_meshes_as_batch(batch, include_textures=True)
    else:
        # try to stack anyway in case the object implements stacking.
        return torch.stack(batch, 0)

def worker_init_fn(i):
    info = get_worker_info()
    if hasattr(info.dataset, 'conf'):
        conf = info.dataset.conf
        set_seed(info.id + conf.seed)
        set_num_threads(conf.num_threads)
    else:
        set_num_threads(1)
# def worker_init_fn(worker_id):
#     info = get_worker_info()
#     if hasattr(info.dataset, 'conf'):
#         conf = info.dataset.conf

#         if dist.is_initialized():
#             rank = dist.get_rank()
#         else:

#         worker_seed = conf.seed + rank * 10000 + worker_id
#         set_seed(worker_seed)
#         print(f"[Rank {rank}] Worker {worker_id} using seed {worker_seed}")

#         set_num_threads(conf.num_threads)
#     else:
#         set_num_threads(1)

class BaseDataset(metaclass=ABCMeta):
    base_default_conf = {
        'name': '???',
        'workers_per_gpu': '???',
        'train_batch_size_per_gpu': '???',
        'val_batch_size_per_gpu': '???',
        'test_batch_size_per_gpu': '???',
        'batch_size_per_gpu': 1,
        'shuffle_training': True,
        'num_threads': 1,
        'seed': 0,
    }
    default_conf = {}
    strict_conf = True

    def __init__(self, conf):
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf))
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        logger.info(f'Creating dataset {self.__class__.__name__}')
        self._init(self.conf)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class."""
        raise NotImplementedError

    def get_data_loader(self, split, shuffle=None, pinned=True,
                        distributed=False):
        """Return a data loader for a given split."""
        assert split in ['train', 'val', 'test']
        dataset = self.get_dataset(split)
        try:
            batch_size = self.conf[split+'_batch_size_per_gpu']
        except omegaconf.MissingMandatoryValue:
            batch_size = self.conf.batch_size_per_gpu
        num_workers = self.conf.get('workers_per_gpu', batch_size)
        if False: # distributed:
            shuffle = False
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
            if shuffle is None:
                shuffle = (split == 'train' and self.conf.shuffle_training)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, pin_memory=pinned, collate_fn=collate,
            num_workers=num_workers, worker_init_fn=worker_init_fn)

    def get_overfit_loader(self, split):
        """Return an overfit data loader.
        The training set is composed of a single duplicated batch, while
        the validation and test sets contain a single copy of this same batch.
        This is useful to debug a model and make sure that losses and metrics
        correlate well.
        """
        assert split in ['train', 'val', 'test']
        dataset = self.get_dataset('train')
        sampler = LoopSampler(
            self.conf.batch_size,
            len(dataset) if split == 'train' else self.conf.batch_size)
        num_workers = self.conf.get('num_workers', self.conf.batch_size)
        return DataLoader(dataset, batch_size=self.conf.batch_size,
                          pin_memory=True, num_workers=num_workers, collate_fn=collate,
                          sampler=sampler, worker_init_fn=worker_init_fn)