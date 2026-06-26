import pickle
import numpy as np
import os.path as osp

import torch
import cv2
import imgaug.augmenters as iaa

def read_template_data(obj_names, paths):
    num_sample_contour_points = {}
    template_views = {}
    orientations = {}
    template_vertices = {}
    for obj_name, path in zip(obj_names, paths):
        with open(path, "rb") as pkl_handle:
            pre_render_dict = pickle.load(pkl_handle)
        head = pre_render_dict['head']
        num_sample_contour_points[obj_name] = head['num_sample_contour_point']
        template_views[obj_name] = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
        orientations[obj_name] = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)
        if 'template_vertex' in pre_render_dict:
            template_vertices[obj_name] = torch.from_numpy(pre_render_dict['template_vertex']).type(torch.float32)

    return num_sample_contour_points, template_views, orientations, template_vertices

def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image

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

def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()

def torch_image_to_numpy(aug_image_ref_single):
    aug_image_ref_single = aug_image_ref_single.permute(1, 2, 0)
    aug_image_ref_single = aug_image_ref_single.cpu().numpy()
    if aug_image_ref_single.dtype != np.uint8:
        aug_image_ref_single = (aug_image_ref_single * 255).astype(np.uint8)
    return aug_image_ref_single

def get_imgaug_seq():
    seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
    ], random_order=True)  # apply augmenters in random order

    return seq
