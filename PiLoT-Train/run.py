"""Hydra entry point. Folder name has hyphens, so we register a valid package alias."""
import sys
import types
from pathlib import Path

import hydra
from omegaconf import DictConfig

_ROOT = Path(__file__).resolve().parent
_PKG = 'pilot_train'


def _bootstrap_package():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_ROOT)]
        pkg.__file__ = str(_ROOT / '__init__.py')
        sys.modules[_PKG] = pkg

    main = sys.modules.get('__main__')
    run_path = str(_ROOT / 'run.py')
    if main is not None and getattr(main, '__file__', None) == run_path:
        sys.modules[f'{_PKG}.run'] = main
        main.__package__ = _PKG


_bootstrap_package()


def train(cfg):
    from .tools.train import train as run_train
    run_train(cfg)


@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == '__main__':
    main()
