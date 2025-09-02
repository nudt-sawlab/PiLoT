from pathlib import Path
import numpy as np
import cv2
# TODO: consider using PIL instead of OpenCV as it is heavy and only used here
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..utils import  undistort
from ..geometry import Camera, Pose

def visualize_image_alignment(img1, img2, img3, title1="Image 1", title2="Image 2", overlay_alpha=0.5, save_path=None):
    """
    Visualize the alignment between two images and optionally save the comparison.

    Args:
        img1: First image (numpy array, BGR or grayscale).
        img2: Second image (numpy array, same dimensions as img1).
        title1: Title for the first image.
        title2: Title for the second image.
        overlay_alpha: Transparency for the overlay visualization (0 to 1).
        save_path: Path to save the visualization (optional).

    Returns:
        None. Displays and optionally saves the visualizations.
    """
    # Ensure the images have the same size
    assert img1.shape[:2] == img2.shape[:2], "Both images must have the same dimensions."

    # Convert images to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    if len(img3.shape) == 3:
        img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    else:
        img3_gray = img3

    # Compute difference image
    diff_12 = cv2.absdiff(img1_gray, img2_gray)

    diff_32 = cv2.absdiff(img3_gray, img2_gray)

    # Create overlay visualization
    overlay_12 = cv2.addWeighted(img1_gray, overlay_alpha, img2_gray, 1 - overlay_alpha, 0)

    overlay_32 = cv2.addWeighted(img3_gray, overlay_alpha, img2_gray, 1 - overlay_alpha, 0)


    # Plot the results
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))

    # Original images
    axs[0, 0].imshow(img1_gray, cmap='gray')
    axs[0, 0].set_title("initial pose")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img2_gray, cmap='gray')
    axs[0, 1].set_title("query")
    axs[0, 1].axis('off')

    # Difference visualization
    axs[1, 0].imshow(img3_gray, cmap='gray')
    axs[1, 0].set_title("refined pose")
    axs[1, 0].axis('off')

    # Overlay visualization
    axs[1, 1].imshow(img2_gray, cmap='gray')
    axs[1, 1].set_title("query")
    axs[1, 1].axis('off')

    # Difference visualization
    axs[2, 0].imshow(diff_12, cmap='hot')
    axs[2, 0].set_title("Difference (Absolute) with initial pose")
    axs[2, 0].axis('off')

    # Overlay visualization
    axs[2, 1].imshow(diff_32, cmap='hot')
    axs[2, 1].set_title("Difference (Absolute) with refined pose")
    axs[2, 1].axis('off')

    # Difference visualization
    axs[3, 0].imshow(overlay_12, cmap='gray')
    axs[3, 0].set_title("Overlay with initial pose")
    axs[3, 0].axis('off')

    # Overlay visualization
    axs[3, 1].imshow(overlay_32, cmap='gray')
    axs[3, 1].set_title("Overlay with refined pose")
    axs[3, 1].axis('off')
    # Adjust layout
    plt.tight_layout()

    # Save to file if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # Show the visualization
    # plt.show()
def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()


def read_image(path, grayscale=False, scale = None, distortion =None, query_camera=None):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if distortion:
        image= undistort.main(image, query_camera, distortion)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    if scale:
        height, width = image.shape[:2]
        new_dim = (int(width // scale), int(height // scale))
        image = cv2.resize(image, new_dim)
    
    return image
def read_image_list(path_list, grayscale=False, scale = None, distortion = None, query_camera = None):
    image_list = []
    for path in tqdm(path_list):
        image = read_image(path, scale = scale, distortion=distortion, query_camera=query_camera)
        image_list.append(image)
    return image_list
def read_render_image_list(path_list, grayscale=False, scale = None):
    image_list = []
    depth_list = []
    
    for path in path_list:
        image_list.append(cv2.imread(str(path), cv2.IMREAD_COLOR))
        depth_list.append(np.load(path.replace('.png', '.npy')))
    
    return image_list, depth_list
def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def crop(image, size, *, random=True, other=None, camera=None,
         return_bbox=False, centroid=None):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    if random:
        top = np.random.randint(0, h - h_new + 1)
        left = np.random.randint(0, w - w_new + 1)
    elif centroid is not None:
        x, y = centroid
        top = np.clip(int(y) - h_new // 2, 0, h - h_new)
        left = np.clip(int(x) - w_new // 2, 0, w - w_new)
    else:
        top = left = 0

    image = image[top:top+h_new, left:left+w_new]
    ret = [image]
    if other is not None:
        ret += [other[top:top+h_new, left:left+w_new]]
    if camera is not None:
        ret += [camera.crop((left, top), (w_new, h_new))]
    if return_bbox:
        ret += [(top, top+h_new, left, left+w_new)]
    return ret


def zero_pad(size, *images):
    ret = []
    for image in images:
        h, w = image.shape[:2]
        padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret


def read_view(conf, image_path: Path, camera: Camera, T_w2cam: Pose,
              p3D: np.ndarray, p3D_idxs: np.ndarray, *,
              rotation=0, random=False):

    img = read_image(image_path, conf.grayscale)
    img = img.astype(np.float32)
    name = image_path.name

    # we assume that the pose and camera were already rotated during preprocess
    if rotation != 0:
        img = np.rot90(img, rotation)

    if conf.resize:
        scales = (1, 1)
        if conf.resize_by == 'max':
            img, scales = resize(img, conf.resize, fn=max)
        elif (conf.resize_by == 'min' or
                (conf.resize_by == 'min_if'
                    and min(*img.shape[:2]) < conf.resize)):
            img, scales = resize(img, conf.resize, fn=min)
        if scales != (1, 1):
            camera = camera.scale(scales)

    if conf.crop:
        if conf.optimal_crop:
            p2D, valid = camera.world2image(T_w2cam * p3D[p3D_idxs])
            p2D = p2D[valid].numpy()
            centroid = tuple(p2D.mean(0)) if len(p2D) > 0 else None
            random = False
        else:
            centroid = None
        img, camera, bbox = crop(
            img, conf.crop, random=random,
            camera=camera, return_bbox=True, centroid=centroid)
    elif conf.pad:
        img, = zero_pad(conf.pad, img)
        # we purposefully do not update the image size in the camera object

    data = {
        'name': name,
        'image': numpy_image_to_torch(img),
        'camera': camera.float(),
        'T_w2cam': T_w2cam.float(),
    }
    return data
