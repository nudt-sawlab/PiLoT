"""Load COLMAP sparse models (text format) for CityGaussian pose alignment."""

import os
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def detect_sparse_dir(data_path: str, sparse_subdir: str = "") -> str:
    """Return the directory containing COLMAP cameras/images files."""
    if sparse_subdir:
        candidate = os.path.join(data_path, sparse_subdir)
        if os.path.isfile(os.path.join(candidate, "images.txt")):
            return candidate
        raise FileNotFoundError(f"COLMAP sparse dir not found: {candidate}")

    candidates = [
        os.path.join(data_path, "sparse", "sparse", "0"),
        os.path.join(data_path, "sparse", "0"),
        os.path.join(data_path, "sparse"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "images.txt")):
            return candidate
    raise FileNotFoundError(f"No COLMAP images.txt found under {data_path}")


def load_colmap_c2w(sparse_dir: str) -> Tuple[np.ndarray, List[str]]:
    """Load all camera-to-world matrices and image names from COLMAP text."""
    from internal.utils.colmap import read_images_text

    images = read_images_text(os.path.join(sparse_dir, "images.txt"))
    c2w_list = []
    image_names = []
    for _, image in images.items():
        rot = R.from_quat([
            image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0],
        ]).as_matrix()
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rot
        w2c[:3, 3] = image.tvec
        c2w_list.append(np.linalg.inv(w2c))
        image_names.append(image.name)
    return np.stack(c2w_list, axis=0), image_names


def compute_similarity_transform(sparse_dir: str) -> np.ndarray:
    """Compute the same scene normalization used by CityGaussian render tools."""
    from internal.utils.normalize import similarity_from_cameras

    c2w_all, _ = load_colmap_c2w(sparse_dir)
    return similarity_from_cameras(c2w_all)


def load_reference_c2w_model(
    sparse_dir: str,
    similarity: np.ndarray,
    reference_image: str,
) -> np.ndarray:
    """Load a reference camera pose in normalized model coordinates."""
    from internal.utils.normalize import transform_cameras

    c2w_all, image_names = load_colmap_c2w(sparse_dir)
    if reference_image not in image_names:
        raise ValueError(
            f"Reference image {reference_image!r} not found in COLMAP sparse model"
        )
    idx = image_names.index(reference_image)
    return transform_cameras(similarity, c2w_all[idx:idx + 1])[0]
