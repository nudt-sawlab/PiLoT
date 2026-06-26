
import torch
import torch.nn as nn
import numpy as np
import cv2
import math

from .base_model import BaseModel
from ..models import get_model
from ..utils.utils import masked_mean, get_closest_template_view_index
from ..utils.geometry.losses import scaled_barron, error_add, error_add_s
from ..utils.transform import  visualize_points_on_images

from .GATs import GraphAttentionLayer

def skew_symmetric(v):
    """Return skew-symmetric matrices for batched 3D vectors."""
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M

@torch.jit.script
def transform_p3d(body2view_pose_data, p3d):
    """Apply rigid transform to 3D points in body coordinates."""
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    t = body2view_pose_data[..., 9:]
    return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2)

@torch.jit.script
def rotate_p3d(body2view_pose_data, p3d):
    """Apply only rotation component of pose to 3D points."""
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    return p3d @ R.transpose(-1, -2)

@torch.jit.script
def project_p3d(camera_data, p3d):
    """Project 3D points to image plane and return visibility mask."""
    eps=1e-4

    z = p3d[..., -1]
    valid1 = z > eps
    z = z.clamp(min=eps)
    p2d = p3d[..., :-1] / z.unsqueeze(-1)

    f = camera_data[..., 2:4]
    c = camera_data[..., 4:6]
    p2d = p2d * f.unsqueeze(-2) + c.unsqueeze(-2)

    size = camera_data[..., :2]
    size = size.unsqueeze(-2)
    valid2 = torch.logical_and(p2d >= 0, p2d <= (size - 1))
    valid2 = torch.logical_and(valid2[..., 0], valid2[..., 1])
    valid = torch.logical_and(valid1, valid2)

    return p2d, valid

class PiLoT(BaseModel):
    default_conf = {
        'success_thresh': 5,
        'clamp_error': 50,
        'normalize_features': True,
        'normalize_cost': True,
        'down_sample_image_mode': 'bilinear',
        'train_score': False,
        'with_refine': False,
        'train_GAT': True,

        'function_length': 8,
        'distribution_length': 16,
        'function_slope': 0.0,
        'function_amplitude': 0.36,
        'min_continuous_distance': 6.0,
        'learning_rate': 1.3,
        'alternative_optimizing': False,
        'cost_fn': 'scaled_barron(0, 0.1)',
        'multi_optimizer': False,

    }

    required_data_keys = {

    }

    strict_conf = False  # need to pass new confs to children models
    eps = 1e-5

    def _init(self, conf):
        """Initialize feature extractor, optimizer, and optional scoring heads."""
        self.conf = conf
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        if conf.multi_optimizer:
            self.optimizer = nn.ModuleList()
            for i in range(len(self.conf.extractor.output_dim)):
                self.optimizer.append(get_model(conf.optimizer.name)(conf.optimizer))
        else:
            self.optimizer = get_model(conf.optimizer.name)(conf.optimizer)

        
        self.GATs = nn.ModuleList()
        for output_dim in (self.conf.extractor.output_dim):
            if output_dim % 4 != 0:
                self.GATs.append(GraphAttentionLayer(output_dim+1, output_dim+1, 0.6, 0.2, concat=False))
            else:
                self.GATs.append(GraphAttentionLayer(output_dim, output_dim, 0.6, 0.2, concat=False))

        if self.conf.train_score:
            inp_dim = len(self.conf.extractor.output_dim) * self.conf.extractor.output_dim[0]
            self.prob_net = nn.Sequential(
                nn.Linear(inp_dim, inp_dim),
                nn.ReLU(),
                nn.Linear(inp_dim, inp_dim),
                nn.ReLU(),
                nn.Linear(inp_dim, 1),
                nn.Sigmoid()
            )

    @torch.no_grad()
    def visualize_uncertainties(self, uncertainties, images):
        """Overlay uncertainty maps on RGB images for debugging."""
        
        display_uncertainty_maps = []
        image_numpy = images.permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
        for i, uncertainty in enumerate(uncertainties):
            uncertainty_numpy = uncertainty.mul(255).byte().cpu().numpy()
            uncertainty_maps = None
            for j, uncertainty_map in enumerate(uncertainty_numpy):
                tmp_uncertainty_map = uncertainty_map[0]
                display_image = image_numpy[j].copy()
                hh, ww = display_image.shape[:2]
                tmp_uncertainty_map = cv2.resize(tmp_uncertainty_map, (ww, hh))
                tmp_uncertainty_map = cv2.applyColorMap(tmp_uncertainty_map, cv2.COLORMAP_JET)
                display_image = cv2.addWeighted(display_image, 0.6, tmp_uncertainty_map, 0.4, 0)
                if uncertainty_maps is None:
                    uncertainty_maps = display_image[None]
                else:
                    uncertainty_maps = np.append(uncertainty_maps,  display_image[None], axis=0)
            display_uncertainty_maps.append(uncertainty_maps)
        return display_uncertainty_maps

    @torch.no_grad()
    def visualize(self, data):
        """Attach initial/optimized/reference visualization images into data dict."""
        data['d_init_images'] = self.visualize_vertex(data['aligned_vertex'], data['d_init_body2view_pose'], data['cameras'], data['images'])
        data['d_opt_images'] = self.visualize_vertex(data['aligned_vertex'], data['opt_body2view_pose'][-1], data['cameras'], data['images'], gt_pose=data['gt_body2view_poses'])
        data['d_ref_images'] = self.visualize_vertex(data['aligned_vertex'], data['gt_body2view_poses_ref'], data['cameras_ref'], data['images_ref'])

    def interpolate_feature_map(self, feature, p2d, return_gradients=False):
        """Sample feature map at projected points and optionally estimate Jacobians."""
        interpolation_pad = 4
        b, c, h, w = feature.shape
        scale = torch.tensor([w-1, h-1]).to(p2d)
        pts = (p2d / scale) * 2  - 1
        pts = pts.clamp(min=-2, max=2)
        fp = torch.nn.functional.grid_sample(feature, pts[:, None], mode='bilinear', align_corners=True)
        fp = fp.reshape(b, c, -1).transpose(-1, -2)
        
        image_size_ = torch.tensor([w-interpolation_pad-1, h-interpolation_pad-1]).to(pts)
        valid = torch.all((p2d >= interpolation_pad) & (p2d <= image_size_), -1)

        if return_gradients:
            dxdy = torch.tensor([[1, 0], [0, 1]])[:, None].to(pts) / scale * 2
            dx, dy = dxdy.chunk(2, dim=0)
            pts_d = torch.cat([pts-dx, pts+dx, pts-dy, pts+dy], 1)
            tensor_d = torch.nn.functional.grid_sample(
                    feature, pts_d[:, None], mode='bilinear', align_corners=True)
            tensor_d = tensor_d.reshape(b, c, -1).transpose(-1, -2)
            tensor_x0, tensor_x1, tensor_y0, tensor_y1 = tensor_d.chunk(4, dim=1)
            gradients = torch.stack([
                (tensor_x1 - tensor_x0)/2, (tensor_y1 - tensor_y0)/2], dim=-1)
        else:
            gradients = torch.zeros(b, pts.shape[1], c, 2).to(feature)

        return fp, valid, gradients

    def run_photometirc_constraint_fusion(self, vertices, pose_ref, pose_q, 
                                          feature_ref, feature_q, cam_ref, cam_q,
                                          uncertainty_q, uncertainty_ref):
        """Compute photometric residual, gradient, and Hessian for fusion update."""
        from ..utils.geometry import losses
        loss_fn = eval('losses.' + self.conf.cost_fn)

        p3d_ref = pose_ref.transform(vertices)
        p2d_ref, visible_ref = cam_ref.view2image(p3d_ref)
        fp_ref, valid_ref, _ = self.interpolate_feature_map(feature_ref, p2d_ref)
        valid_ref = valid_ref & visible_ref
        weight_ref, _, _ = self.interpolate_feature_map(uncertainty_ref, p2d_ref)

        p3d_q = pose_q.transform(vertices)
        p2d_q, visible_q = cam_q.view2image(p3d_q)
        fp_q, valid_q, J_f = self.interpolate_feature_map(feature_q, p2d_q, return_gradients=True)
        valid_q = valid_q & visible_q
        weight_q, _, _ = self.interpolate_feature_map(uncertainty_q, p2d_q, return_gradients=False)

        res = fp_q - fp_ref
        J_p3d_pose = pose_q.R[:, None] @ pose_q.J_transform(vertices)
        J_p2d_p3d, _ = cam_q.J_world2image(p3d_q)
        J = J_f @ J_p2d_p3d @ J_p3d_pose

        valid = valid_q & valid_ref
        cost = (res**2).sum(-1)
        cost, w_loss, _ = loss_fn(cost)
        weight = w_loss * valid.float() * weight_q[..., 0] * weight_ref[..., 0]

        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weight[..., None] * grad
        grad = grad.sum((-2))

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weight[..., None, None] * Hess
        Hess = Hess.sum((-3))

        return -grad.unsqueeze(-1), Hess, cost.mean(dim=-1)
    @torch.no_grad()
    def visualize_aero(self, data, name = None):
        """Visualize init/optimized/GT correspondences for Aero sequence."""
        init_pose = data['d_init_body2view_pose']
        gt_body2view_pose = data['gt_body2view_poses']

        pose = data['opt_body2view_pose'][-1]
        vertex_in_body = data['aligned_vertex']
        camera_q = data['cameras']
        image = data['images']
        camera_ref = data['cameras_ref']

        gt_body2view_poses_ref = data['gt_body2view_poses_ref']


        p3d_query_gt = gt_body2view_pose.transform(vertex_in_body)
        p2d_gt, visible_2d = camera_q.view2image(p3d_query_gt)
        p2d_gt = p2d_gt.squeeze(0).cpu().numpy()

        p3d_query_init = init_pose.transform(vertex_in_body)
        p2d, visible_2d = camera_q.view2image(p3d_query_init)
        p2d = p2d.squeeze(0).cpu().numpy()

        p3d_query_opt = pose.transform(vertex_in_body)
        p2d_opt, visible_2d = camera_q.view2image(p3d_query_opt)
        p2d_opt = p2d_opt.squeeze(0).cpu().numpy()

        p3d_ref = gt_body2view_poses_ref.transform(vertex_in_body)
        p2d_ref, visible_2d_ref = camera_ref.view2image(p3d_ref)
        data['prob_optimizing_result_images'] = []
        B = 0
        if name is not None:
            B = data['output_name'].index(name)
        if len(p2d.shape) == 2:
            p2d = np.expand_dims(p2d,axis=0)
            p2d_gt = np.expand_dims(p2d_gt,axis=0)
            p2d_opt = np.expand_dims(p2d_opt,axis=0)

        img = data['images'][B].cpu().numpy()
        img_ref = data['images_ref'][B].cpu().numpy()
        save_path = 'verify'
        num = 30
        indices = np.random.choice(len(p2d[B]), size=num, replace=False)
        
        display_images1 = visualize_points_on_images(img, img_ref, 
                    p2d[B, indices],
                    p2d_ref[B, indices],
                    save_path = save_path, extra = 'initial')
        display_images2 = visualize_points_on_images(img, img_ref, 
                    p2d_opt[B, indices],
                    p2d_ref[B, indices],
                    save_path = save_path, extra = 'opt')

        display_images3 = visualize_points_on_images(img, img_ref, 
                    p2d_gt[B, indices],
                    p2d_ref[B, indices],
                    save_path = save_path, extra = 'gt')
        combined_img = cv2.vconcat([display_images1, display_images2, display_images3])
        data['vis_gt_imlg']= combined_img
    def extract_score(self, vertices, pose_ref, pose_q, 
                      feature_ref, feature_q, cam_ref, cam_q, GAT=None):
        from ..utils.geometry import losses
        loss_fn = eval('losses.' + self.conf.cost_fn)

        num_vertex = vertices.shape[-2]
        batch_size, num_init_pose, num_ref = pose_ref.shape[:3]
        p3d_ref = pose_ref.transform(vertices[:, :, None])
        p2d_ref, visible_ref = cam_ref.view2image(p3d_ref)
        fp_ref, valid_ref, _ = self.interpolate_feature_map(feature_ref.flatten(0, 2), p2d_ref.flatten(0, 2))
        fp_ref = fp_ref.view(batch_size, num_init_pose, num_ref, *fp_ref.shape[1:])
        valid_ref = valid_ref.view(batch_size, num_init_pose, num_ref, *valid_ref.shape[1:])
        valid_ref = valid_ref & visible_ref

        if GAT is None:
            fp_ref_avg = (valid_ref.float()[..., None] * fp_ref).sum(2) / (valid_ref.float().sum(2)[..., None]).clamp(min=self.eps)
        else:
            if self.conf.extractor.output_dim[0] % 4 != 0:
                raise NotImplementedError
                tmp_ones = torch.ones((*fp_ref.shape[:-1], 1), device=fp_ref.device)
                fp_ref_inp = torch.cat((fp_ref, tmp_ones), dim=-1)
                fp_ref_inp[:, 0, :, -1] = 0
                fp_ref_avg = GAT(fp_ref_inp, valid_ref)
                fp_ref_avg = fp_ref_avg[..., :-1]
            else:
                fp_ref_avg = GAT(fp_ref.flatten(0, 1), valid_ref.flatten(0, 1))
                fp_ref_avg = fp_ref_avg.view(batch_size, num_init_pose, *fp_ref_avg.shape[1:])

        p3d_q = pose_q.transform(vertices[:, :, None])
        p2d_q, visible_q = cam_q.unsqueeze(1).unsqueeze(1).view2image(p3d_q)
        fp_q, valid_q, J_f = self.interpolate_feature_map(feature_q, p2d_q.flatten(1, 3), return_gradients=True)
        fp_q = fp_q.view(batch_size, num_init_pose, -1, num_vertex, fp_q.shape[-1])
        valid_q = valid_q.view(batch_size, num_init_pose, -1, num_vertex)
        J_f = J_f.view(batch_size, num_init_pose, -1, num_vertex, *J_f.shape[-2:])

        res = fp_q - fp_ref_avg[:, :, None]
        
        return res

    def run_photometirc_constraint(self, vertices, pose_r, pose_q, f_r, f_q, cam_r, cam_q):
        """Compute single-view photometric Gauss-Newton terms."""
        from ..utils.geometry import losses
        loss_fn = eval('losses.' + self.conf.cost_fn)

        p2d_r, visible_r = cam_r.view2image(pose_r.transform(vertices))
        p2d_q, visible_q = cam_q.view2image(pose_q.transform(vertices))
        fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)
        fp_q, valid_q, J_f = self.interpolate_feature_map(f_q, p2d_q, return_gradients=True)

        valid = (visible_r & visible_q & valid_r & valid_q).detach()

        res = fp_q - fp_r
        J_p3d_pose = pose_q.R[:, None] @ pose_q.J_transform(vertices)
        J_p2d_p3d, _ = cam_q.J_world2image(pose_q.transform(vertices))
        J = J_f @ J_p2d_p3d @ J_p3d_pose

        cost = (res**2).sum(-1)
        cost, w_loss, _ = loss_fn(cost)
        weight = w_loss * valid.float()

        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weight[..., None] * grad
        grad = grad.sum((1))

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weight[..., None, None] * Hess
        Hess = Hess.sum((1))

        return -grad.unsqueeze(-1), Hess


    def extract_pose_score(self, image, feature, init_body2view_pose_data, camera_data, template_view, fore_hist, back_hist):
        """Estimate per-pose contour confidence score."""
        normals_in_image, centers_in_image, centers_in_body, \
        lines_image_pf_segments, lines_image_pb_segments, valid_data_line, lines_amplitude, lines_slop, lines_feature = \
            self.contour_feature_map_extractor.forward(image, feature, init_body2view_pose_data, camera_data, template_view, fore_hist, back_hist)
        
        distributions, distribution_mean, distribution_uncertainties =\
            self.boundary_predictor.forward(lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude)
        
        score = (1 - distribution_mean.abs() / self.distribution_length_minus_1_half) * distribution_uncertainties
        score = score.sum(-1)

        return score 

    def _forward(self, data, visualize=False, tracking=False):
        """Run multi-scale Aero pose refinement and collect training outputs."""
        images = data['images']
        device = images.device
        batch_size, C, H, W = images.shape
        images_ref = data['images_ref']
        images_input = torch.cat((images, images_ref), dim=0)
        feature_maps_output, uncertainties_output = self.extractor._forward(images_input)
        
        if self.conf.normalize_features:
            def normalize_features(fl):
                for i, feature in enumerate(fl):
                    fl[i] = torch.nn.functional.normalize(feature, dim=1)
            normalize_features(feature_maps_output)
        feature_maps = []
        feature_maps_ref = []
        uncertainties_q = []
        uncertainties_ref = []

        for i, feature_output in enumerate(feature_maps_output):
            feature_maps.append(feature_output[:images.shape[0]])
            feature_maps_ref.append(feature_output[images.shape[0]:])
            uncertainties_q.append(uncertainties_output[i][:images.shape[0]])
            uncertainties_ref.append(uncertainties_output[i][images.shape[0]:])

        vertex_in_body = data['aligned_vertex']
        camera_ref = data['cameras_ref']
        gt_body2view_poses = data['gt_body2view_poses']
        gt_body2view_poses_ref = data['gt_body2view_poses_ref']
        init_body2view_poses = data['init_body2view_poses']
        deformed_obj_pose_detection = init_body2view_poses
        camera_q = data['cameras']
        gt_p2d, gt_valid = camera_q.view2image(gt_body2view_poses.transform(vertex_in_body))

        data['d_init_body2view_pose'] = deformed_obj_pose_detection
        data['opt_body2view_pose'] = []
        data['err_reprojection'] = []
 
        for i, s in enumerate(self.conf.scales):
            constraint = self.conf.constraints[i]
            if self.conf.multi_optimizer:
                optimizer = self.optimizer[s]
            else:
                optimizer = self.optimizer

            image_scale = float(2 ** s)
            camera_pyr_q = camera_q.scale(1 / image_scale)
            camera_pyr_ref = camera_ref.scale(1 / image_scale)

            feature_q = feature_maps[-(s+1)]
            feature_ref = feature_maps_ref[-(s+1)]
            uncertainty_q = uncertainties_q[-(s+1)]
            uncertainty_ref = uncertainties_ref[-(s+1)]

            B, A, cost = \
                self.run_photometirc_constraint_fusion(vertex_in_body, 
                                                        gt_body2view_poses_ref, 
                                                        deformed_obj_pose_detection,
                                                        feature_ref, 
                                                        feature_q, 
                                                        camera_pyr_ref, 
                                                        camera_pyr_q, uncertainty_q, uncertainty_ref)
            optimizing_pose_q = optimizer(dict(pose=deformed_obj_pose_detection, B=B, A=A))
            data['opt_body2view_pose'].append(optimizing_pose_q)
            deformed_obj_pose_detection = optimizing_pose_q.detach()

            p2ds, valids = \
                camera_q.view2image(optimizing_pose_q.transform(vertex_in_body))
            err = torch.sum((gt_p2d - p2ds) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, gt_valid, -1)
            err = err.view(batch_size, -1).clamp(max=self.conf.clamp_error)
            data['err_reprojection'].append(err)
        # self.visualize_aero(data)

        return data

    def loss(self, pred, data):
        """Compute reprojection loss across optimization scales."""
        
        losses = {'total': 0.}

        success = None
        for i, err_reprojection in enumerate(data['err_reprojection']):
            
            scale = len(data['err_reprojection']) - i
            thresh = self.conf.success_thresh * scale
            success = err_reprojection < thresh
            if i==0:
                success[...] = True
            loss_reprojection = ((err_reprojection * success.float())).sum(dim=-1) / (success.sum(dim=-1)+self.eps)
            losses[f'loss_reprojection/{i}'] = loss_reprojection
            losses['total'] += loss_reprojection / len(data['err_reprojection'])

        return losses

    def metrics(self, pred, data):
        """Compute pose and ADD(-S) metrics for evaluation."""
        metrics = {'R_error': [], 't_error': [], 'err_add': [], 'err_add_s': [],
                   'err_add(s)': [], 'err_add_init': [], 'err_add_s_init': [], 'err_add(s)_init': []}  # = self.loss(pred, data)
        vertices = pred['aligned_vertex']

        def scaled_pose_error(body2view_pose, gt_body2view_pose, gt_view2body_pose):
            err_t = torch.norm(body2view_pose.t - gt_body2view_pose.t, dim=-1)
            err_R = torch.acos((((body2view_pose.R @ gt_view2body_pose.R)
                                .diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1))
            err_R = torch.rad2deg(err_R)
            return err_R, err_t

        gt_view2body_pose = pred['gt_body2view_poses'].inv()
        gt_body2view_pose = pred['gt_body2view_poses']
        init_body2view_pose = pred['d_init_body2view_pose']
        opt_body2view_pose = pred['opt_body2view_pose'][-1]
        with torch.no_grad():
            R_error, t_error = scaled_pose_error(opt_body2view_pose, gt_body2view_pose, gt_view2body_pose)
            metrics['R_error'].append(R_error)
            metrics['t_error'].append(t_error)
            m_err_add = error_add(vertices, gt_body2view_pose, opt_body2view_pose)
            m_err_add_s = error_add_s(vertices, gt_body2view_pose, opt_body2view_pose)
            metrics['err_add'].append(m_err_add)
            metrics['err_add_s'].append(m_err_add_s)
            m_err_add_1s1 = m_err_add
            m_err_add_1s1[pred['sysmetric']] = m_err_add_s[pred['sysmetric']]
            metrics['err_add(s)'].append(m_err_add_1s1)
            init_err_add = error_add(vertices, gt_body2view_pose, init_body2view_pose)
            init_err_add_s = error_add_s(vertices, gt_body2view_pose, init_body2view_pose)
            metrics['err_add_init'].append(init_err_add)
            metrics['err_add_s_init'].append(init_err_add_s)
            init_err_add_1s1 = init_err_add
            init_err_add_1s1[pred['sysmetric']] = init_err_add_s[pred['sysmetric']]
            metrics['err_add(s)_init'].append(init_err_add_1s1)
        metrics['R_error'] = torch.stack(metrics['R_error']).view(-1)
        metrics['t_error'] = torch.stack(metrics['t_error']).view(-1)
        metrics['err_add'] = torch.stack(metrics['err_add']).view(-1)
        metrics['err_add_s'] = torch.stack(metrics['err_add_s']).view(-1)
        metrics['err_add(s)'] = torch.stack(metrics['err_add(s)']).view(-1)
        metrics['err_add_init'] = torch.stack(metrics['err_add_init']).view(-1)
        metrics['err_add_s_init'] = torch.stack(metrics['err_add_s_init']).view(-1)
        metrics['err_add(s)_init'] = torch.stack(metrics['err_add(s)_init']).view(-1)
        # metrics['diameter'] = torch.stack(metrics['diameter']).view(-1)

        return metrics

    def forward_train(self, data):
        pred = self._forward(data)
        losses = self.loss(pred, data)
        metrics = self.metrics(pred, data)

        return pred, losses

    def forward_eval(self, data, visualize, tracking=False):
        pred = self._forward(data, visualize, tracking)
        losses = self.loss(pred, data)
        metrics = self.metrics(pred, data)

        return pred, losses, metrics