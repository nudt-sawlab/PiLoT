"""Minimal GaussianModel for inference only.

Adapted from 3D Gaussian Splatting / Feature 3DGS.
Only the PLY loading and property accessors needed at render time are kept;
all training-related code (optimizer, densification, etc.) is removed.

Original: https://github.com/graphdeco-inria/gaussian-splatting
License: see https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
"""

import numpy as np
import torch
from plyfile import PlyData
from torch import nn


class GaussianModel:

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._semantic_feature = torch.empty(0)
        self.max_radii2D = torch.empty(0)

    # -- properties used by the rasterizer ---------------------------------

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_semantic_feature(self):
        return self._semantic_feature

    # -- PLY I/O -----------------------------------------------------------

    def load_ply(self, path: str) -> None:
        plydata = PlyData.read(path)
        el = plydata.elements[0]

        xyz = np.stack(
            (np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])),
            axis=1,
        )
        opacities = np.asarray(el["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(el["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(el["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(el["f_dc_2"])

        extra_f_names = sorted(
            [p.name for p in el.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(el[attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = sorted(
            [p.name for p in el.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(el[attr_name])

        rot_names = sorted(
            [p.name for p in el.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(el[attr_name])

        sem_names = sorted(
            [n for n in el.data.dtype.names if n.startswith("semantic_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if sem_names:
            semantic_feature = np.stack(
                [np.asarray(el[n]) for n in sem_names], axis=1,
            )
            semantic_feature = np.expand_dims(semantic_feature, axis=-1)
        else:
            semantic_feature = np.zeros((xyz.shape[0], 0, 1))

        def _to_cuda(arr):
            return torch.tensor(arr, dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(
            _to_cuda(xyz).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            _to_cuda(features_dc).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            _to_cuda(features_extra).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            _to_cuda(opacities).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            _to_cuda(scales).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            _to_cuda(rots).requires_grad_(True)
        )
        self._semantic_feature = nn.Parameter(
            _to_cuda(semantic_feature).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
