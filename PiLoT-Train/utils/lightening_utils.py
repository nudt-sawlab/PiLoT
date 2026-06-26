import logging
import os
import time
from omegaconf import OmegaConf
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
# from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
from termcolor import colored
import torch.distributed as dist
import torch

def convert_old_model(old_model_dict):
    if "pytorch-lightning_version" in old_model_dict:
        raise ValueError("This model is not old format. No need to convert!")
    version = pl.__version__
    epoch = old_model_dict["epoch"]
    global_step = old_model_dict["iter"]
    state_dict = old_model_dict["state_dict"]
    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        new_state_dict["model." + name] = value

    new_checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "pytorch-lightning_version": version,
        "state_dict": new_state_dict,
        "lr_schedulers": [],
    }

    if "optimizer" in old_model_dict:
        optimizer_states = [old_model_dict["optimizer"]]
        new_checkpoint["optimizer_states"] = optimizer_states

    return new_checkpoint

def load_model_weight(model, checkpoint, logger):
    state_dict = checkpoint["state_dict"]
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            logger.info("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            logger.info("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

class MyLightningLogger(LightningLoggerBase):
# class MyLightningLogger(Logger):
    def __init__(self, name, save_dir, **kwargs):
        super().__init__()
        self._name = name # "NanoDet"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(save_dir, f"logs-{self._version}")

        self._fs = get_filesystem(save_dir)
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self._init_logger()

        self._experiment = None
        self._kwargs = kwargs

    @property
    def name(self):
        return self._name

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                "the dependencies to use torch.utils.tensorboard "
                "(applicable to PyTorch 1.1 or higher)"
            ) from None

        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def _init_logger(self):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # create file handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        # set file formatter
        f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
        file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # set console formatter
        c_fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def dump_cfg(self, cfg_node, cfg_name):
        with open(os.path.join(self.log_dir, cfg_name), "w") as f:
            # cfg_node.dump(stream=f)
            OmegaConf.save(cfg_node, f)

    @rank_zero_only
    def dump_cfg_with_dir(self, cfg_node, save_dir):
        with open(os.path.join(save_dir, "train_cfg.yml"), "w") as f:
            # cfg_node.dump(stream=f)
            OmegaConf.save(cfg_node, f)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")
        for k, v in metrics.items():
            self.experiment.add_scalars("Val_metrics/" + k, {"Val": v}, step)

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        self.experiment.flush()
        self.experiment.close()
        self.save()

def gather_results(results):
    rank = -1
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    all_results = {}
    for key, value in results.items():
        shape_tensor = torch.tensor(value.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)

        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.float32, device="cuda")
        part_send[: shape_tensor[0]] = value
        part_recv_list = [value.new_zeros(shape_max, dtype=torch.float32) for _ in range(world_size)]
        dist.all_gather(part_recv_list, part_send)

        if rank < 1:
            for recv, shape in zip(part_recv_list, shape_list):
                if key not in all_results:
                    all_results[key] = recv[: shape[0]]
                else:
                    all_results[key] = torch.cat((all_results[key], recv[: shape[0]]))
    return all_results

def rank_filter(func):
    def func_filter(local_rank=-1, *args, **kwargs):
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            pass
    return func_filter

@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)