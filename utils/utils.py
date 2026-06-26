import os
import inspect
import torch
from torch import nn
from .geometry.wrappers import Pose, Camera
from pytorch3d.renderer import (
    PointLights, PerspectiveCameras,BlendParams,
    MeshRasterizer, RasterizationSettings, 
    HardPhongShader, SoftPhongShader, HardGouraudShader, SoftGouraudShader, SoftSilhouetteShader,
    HardFlatShader)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.utils import cameras_from_opencv_projection

def get_file_list(dir, file_list, ext=None):
    new_dir = dir
    if os.path.isfile(dir):
        if ext is None:
            file_list.append(dir)
        else:
            if ext in dir.split('.')[-1]:
                file_list.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir, s)
            get_file_list(new_dir, file_list, ext)

    return file_list

def get_class(mod_name, base_path, base_dir, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
       the module named mod_name, child of base_path.
    """
    file_list = []
    mod_path = None
    get_file_list(base_dir, file_list, 'py')
    for file in file_list:
        file_name = os.path.basename(file).split('.')[0]
        if file_name == mod_name and '__' not in file:
            whole_path = file[:-3].replace('/', '.')
            p = whole_path.find(base_path)
            if p >= 0:
                mod_path = whole_path[p:]
            else:
                mod_path = f'{base_path}.{mod_name}'
            break

    if mod_path is None:
        raise NotImplementedError

    mod = __import__(mod_path, fromlist=[''])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]

def generate_spheric_cameras(sphere_radius, n_divide, device='cpu'):
    import numpy as np
    assert isinstance(sphere_radius, list)
    assert 0 < n_divide <= 4

    sphere_radius = torch.from_numpy(np.stack(sphere_radius))
    n_dist = sphere_radius.shape[0]

    geodesic_points = np.loadtxt('data/template/geodesic_points_%d.txt' % n_divide)
    geodesic_points = torch.from_numpy(geodesic_points) \
        .unsqueeze(0).expand(n_dist, -1, -1).to(device).type(torch.float32)

    downwards = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32) \
        .unsqueeze(0).unsqueeze(0).expand(n_dist, geodesic_points.shape[1], 3)
    view2world_matrix = torch.zeros(size=(n_dist, geodesic_points.shape[1], 4, 4),
                                    device=device, dtype=torch.float32)

    view2world_matrix[..., :3, 3] = geodesic_points * sphere_radius.unsqueeze(-1).unsqueeze(-1)
    view2world_matrix[..., 3, 3] = 1
    view2world_matrix[..., :3, 2] = -geodesic_points
    view2world_matrix[..., :3, 0] = torch.nn.functional.normalize(torch.cross(downwards, -geodesic_points), dim=-1)
    view2world_matrix = view2world_matrix.view(-1, 4, 4)
    view2world_matrix[torch.norm(view2world_matrix[..., :3, 0], dim=-1) == 0, :3, 0] = \
        torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    view2world_matrix = view2world_matrix.view(n_dist, -1, 4, 4)
    view2world_matrix[..., :3, 1] = torch.cross(view2world_matrix[..., :3, 2], view2world_matrix[..., :3, 0])

    return view2world_matrix

# offset_angle: (n) tensor
# offset_translation: (n) tensor
def generate_random_aa_and_t(min_offset_angle, max_offset_angle, min_offset_translation, max_offset_translation):
    if isinstance(min_offset_angle, float):
        min_offset_angle = torch.tensor([min_offset_angle], dtype=torch.float32)
    if isinstance(max_offset_angle, float):
        max_offset_angle = torch.tensor([max_offset_angle], dtype=torch.float32)
    if isinstance(min_offset_translation, float):
        min_offset_translation = torch.tensor([min_offset_translation], dtype=torch.float32)
    if isinstance(max_offset_translation, float):
        max_offset_translation = torch.tensor([max_offset_translation], dtype=torch.float32)

    n = min_offset_angle.shape[0]
    axis = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    angle = (torch.rand(n) * (max_offset_angle - min_offset_angle) + min_offset_angle).unsqueeze(-1) / 180 * 3.1415926
    aa = axis * angle

    direction = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    t = (torch.rand(n) * (max_offset_translation - min_offset_translation) + min_offset_translation).unsqueeze(-1)
    trans = direction * t

    return aa, trans

# p2d: (n, 2) or (b, n, 2)
# return: (4) or (b, 4), [center_x, center_y, w, h]
def get_bbox_from_p2d(p2d):

    bbox_min, _ = torch.min(p2d, dim=-2)
    bbox_max, _ = torch.max(p2d, dim=-2)

    bbox_center = (bbox_min + bbox_max) / 2
    bbox_wh = bbox_max - bbox_min
    bbox = torch.cat((bbox_center, bbox_wh), dim=-1)
    return bbox

def vertex_on_normal_to_image(centers, normals, step):
    return centers + normals * step

def get_closest_template_view_index(body2view_pose: Pose, orientations_in_body):
    orientation = body2view_pose.R.inverse() @ body2view_pose.t.unsqueeze(-1)
    orientation = torch.nn.functional.normalize(orientation, dim=-2).transpose(-1, -2)
    _, index = torch.max(torch.sum(orientation * orientations_in_body, dim=-1), dim=-1)

    return index

def get_closest_k_template_view_index(body2view_pose: Pose, orientations_in_body, k):
    orientation = body2view_pose.R.inverse() @ body2view_pose.t.unsqueeze(-1)
    orientation = torch.nn.functional.normalize(orientation, dim=-2).transpose(-1, -2)
    _, indices = torch.topk(torch.sum(orientation * orientations_in_body, dim=-1), k=k, dim=-1)
    return indices

def project_correspondences_line(template_view, body2view_pose: Pose, camera: Camera, num_sample_center=None):
    if num_sample_center != None:
        step = template_view.shape[1] // num_sample_center
        sample_template_view = template_view[:, ::step, :]
    else:
        sample_template_view = template_view
    centers_in_body = sample_template_view[..., :3]
    normals_in_body = sample_template_view[..., 3:6]
    foreground_distance = sample_template_view[..., 6]
    background_distance = sample_template_view[..., 7]

    centers_in_view = body2view_pose.transform(centers_in_body)
    centers_in_image, centers_valid = camera.view2image(centers_in_view)
    normals_in_view = body2view_pose.rotate(normals_in_body)
    normals_in_image = torch.nn.functional.normalize(normals_in_view[..., :2], dim=-1)

    cur_foreground_distance = foreground_distance * camera.f[..., 0].unsqueeze(-1) / centers_in_view[..., 2]
    cur_background_distance = background_distance * camera.f[..., 0].unsqueeze(-1) / centers_in_view[..., 2]

    data_lines = {'centers_in_body': centers_in_body,
                 'centers_in_view': centers_in_view,
                 'centers_in_image': centers_in_image,
                 'centers_valid': centers_valid,
                 'normals_in_image': normals_in_image,
                 'foreground_distance': cur_foreground_distance,
                 'background_distance': cur_background_distance}

    if torch.any(torch.isnan(data_lines['normals_in_image'])) or torch.any(torch.isnan(data_lines['centers_in_image'])) \
            or torch.any(torch.isnan(data_lines['centers_in_body'])) or torch.any(torch.isnan(data_lines['centers_in_view'])):
            import ipdb;
            ipdb.set_trace();

    return data_lines

def get_lines_image(change_template_view, image, closest_template_views, closest_orientations_in_body,
                    body2view_pose, camera, normal_line_length, num_sample_center=None, mode='nearest'):
    height, width = image.shape[2:]
    if change_template_view:
        index = get_closest_template_view_index(body2view_pose, closest_orientations_in_body)
        template_view = torch.stack([closest_template_views[b][index[b]]
                                     for b in range(closest_template_views.shape[0])])
    else:
        template_view = torch.stack([closest_template_views[b][0]
                                     for b in range(closest_template_views.shape[0])])
    data_lines = project_correspondences_line(template_view, body2view_pose, camera, num_sample_center)
    centers_in_image = data_lines['centers_in_image']
    normals_in_image = data_lines['normals_in_image']
    interpolate_step = torch.arange(-normal_line_length, normal_line_length, device=image.device).unsqueeze(0).unsqueeze(0)\
                           .unsqueeze(-1).expand(centers_in_image.shape[0], centers_in_image.shape[1], -1, -1) + 0.5
    centers = centers_in_image.unsqueeze(2).expand(-1, -1, interpolate_step.shape[2], -1)
    normals = normals_in_image.unsqueeze(2).expand(-1, -1, interpolate_step.shape[2], -1)
    points = centers + interpolate_step * normals
    points[..., 0] = (points[..., 0] / width) * 2 - 1
    points[..., 1] = (points[..., 1] / height) * 2 - 1
    lines_image = torch.nn.functional.grid_sample(image, points, mode=mode, align_corners=False)
    return lines_image, data_lines, template_view

def masked_mean(x, mask, dim, confindence=None):
    mask = mask.float()
    if confindence is not None:
        mask *= confindence
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def checkpointed(cls, do=True):
    '''Adapted from the DISK implementation of Michał Tyszkiewicz.'''
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(
                        super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls

def pack_lr_parameters(params, base_lr, lr_scaling, logger):
    '''Pack each group of parameters with the respective scaled learning rate.
    '''
    from collections import defaultdict

    filters, scales = tuple(zip(*[
        (n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info('Parameters with scaled learning rate:\n{}'.format(
                {s: [n for n, _ in ps] for s, ps in scale2params.items()
                 if s != 1}))
    lr_params = [{'lr': scale*base_lr, 'params': [p for _, p in ps]}
                 for scale, ps in scale2params.items()]
    return lr_params

def init_renderer(device, H, W):
    image_size = (W, H)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        # bin_size=None,
    )
    # self.blend_params = BlendParams(gamma=1e-12, sigma=1e-12, background_color=(0.0, 0.0, 0.0))
    blend_params = BlendParams(gamma=1e-4, sigma=1e-4, background_color=(0.0, 0.0, 0.0))
    image_renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings
        ),
        shader = SoftPhongShader(
            cameras=None,
            blend_params=blend_params,
            device=device
        )
    )

    return image_renderer

@torch.no_grad()
def depth2xyzmap(depths, cameras, poses=None):
    invalid_mask = (depths<0.0)
    B, H, W = depths.shape
    device = depths.device
    
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    vs = vs.reshape(-1).to(device)
    us = us.reshape(-1).to(device)

    zs = depths[:, vs, us]
    xs = (us[None]-cameras.c[..., 0, None])*zs/cameras.f[..., 0, None]
    ys = (vs[None]-cameras.c[..., 1, None])*zs/cameras.f[..., 1, None]
    pts = torch.stack((xs, ys, zs), -1)  #(N,3)
    xyz_map = torch.zeros((B, H, W, 3), dtype=torch.float32, device=device)
    xyz_map[:, vs, us] = pts
    xyz_map[invalid_mask] = 0
    xyz_map = xyz_map.permute(0, 3, 1, 2)

    if poses is not None:
        vertex = poses.inv().transform(pts)
        vertex_map = torch.zeros((B, H, W, 3), dtype=torch.float32, device=device)
        vertex_map[:, vs, us] = vertex
        vertex_map[invalid_mask] = 0
        vertex_map = vertex_map.permute(0, 3, 1, 2)
    else:
        vertex_map = None

    # from ..utils.draw_tutorial import draw_vertices_to_obj
    # draw_vertices_to_obj(xyz_map[0][~invalid_mask[0]].cpu().numpy(), 'src_open/pcd.obj')
    # import ipdb
    # ipdb.set_trace()

    # import ipdb
    # ipdb.set_trace()
    return xyz_map, vertex_map

@torch.no_grad()
def run_render(image_renderer, poses, cameras, image_size, meshes_to_render, default_lights=True, seperate_lights=True):
    device = poses.device
    # default_lights = True
    # seperate_lights = True
    rotations = poses.R
    translations = poses.t

    verts_list = meshes_to_render.verts_list()
    zbuf_list = []
    for verts, pose in zip(verts_list, poses):
        points_3d = pose.transform(verts) # torch.matmul(rotation, verts.transpose(0, 1)) + translation[:, None]
        zbuf_list.append(points_3d[:, -1])
    zbuf = torch.cat(zbuf_list)
    zfar, znear = torch.max(zbuf).item(), torch.min(zbuf).item()
    zfar, znear = (zfar // 100 + 1) * 100., (znear // 100) * 100.
    
    tmp_zeros = torch.zeros_like(cameras.f[..., 0])
    tmp_ones = torch.ones_like(cameras.f[..., 0])
    intrisic_matrix = torch.stack([cameras.f[..., 0], tmp_zeros, cameras.c[..., 0],
                                    tmp_zeros, cameras.f[..., 1], cameras.c[..., 1],
                                    tmp_zeros, tmp_zeros, tmp_ones], dim=-1).reshape(-1, 3, 3).to(device)
    image_size = torch.tensor(image_size, device=device)[None].expand(poses.shape[0], 2)
    cameras_to_render = cameras_from_opencv_projection(poses.R, poses.t, intrisic_matrix, image_size)

    if not default_lights:
        # for ITODD
        if seperate_lights:
            znear_list = torch.stack([z.min() for z in zbuf_list])
            znear_list = torch.maximum(znear_list - 0.4, torch.zeros_like(znear_list))
            loc = torch.stack([torch.zeros_like(znear_list), torch.zeros_like(znear_list), znear_list], axis=-1)
            loc = (rotations @ loc[..., None]).view(-1, 3)
        else:
            loc = torch.tensor((0., 0. ,znear/4)).to(translations.device).view(1,-1,1) # flipped Z
            loc = (rotations@loc).view(-1,3)
        lights = PointLights(diffuse_color=((.5, .5, .5),), ambient_color=((.8, .8, .8),), specular_color=((1., 1., 1.,),), location=loc, device=device)
    else:
        if seperate_lights:
            znear_list = torch.stack([z.min() for z in zbuf_list])
            znear_list = torch.maximum(znear_list - 0.4, torch.zeros_like(znear_list))
            loc = torch.stack([torch.zeros_like(znear_list), torch.zeros_like(znear_list), znear_list], axis=-1)
            loc = (rotations @ loc[..., None]).view(-1, 3)
            lights = PointLights(location=loc, device=device)
        else:
            lights = PointLights(device=device)
    
    rendered_images, rendered_fragments = image_renderer(meshes_to_render, cameras=cameras_to_render, znear=znear, zfar=zfar, lights=lights)
    
    rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
    rendered_depths = rendered_fragments.zbuf
    rendered_depths = rendered_depths[..., 0]
    rendered_masks = (rendered_depths > 0).to(torch.float32)
    rendered_xyz_maps, rendered_vertex_maps = depth2xyzmap(rendered_depths, cameras, poses)
    # rendered_xyz_maps = rendered_xyz_maps.permute(0, 3, 1, 2)
    output = {
        'images': rendered_images,
        'depths': rendered_depths,
        'masks': rendered_masks,
        'xyz_maps': rendered_xyz_maps,
        'vertex_maps': rendered_vertex_maps
    }

    return output