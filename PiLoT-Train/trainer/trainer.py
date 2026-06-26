import torch
import copy
import os
from omegaconf import OmegaConf
import pickle
import warnings
import time
import pytorch_lightning as pl
import torchvision

from typing import Any, List
import torch.distributed as dist
from pytorch_lightning import LightningModule
# from ..models.mobilenetv2_unet import MobileNetV2_unet
from ..dataset.base_dataset import set_seed
from pytorch_lightning import seed_everything
from ..models import get_model
from ..utils.lightening_utils import gather_results, mkdir
from ..utils.utils import pack_lr_parameters

class Trainer(pl.LightningModule):

    def __init__(self, cfg, train_data_loader=None, val_data_loader=None):
        super().__init__()

        self.cfg = cfg
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.save_flag = -10
        self.model = get_model(cfg.models.name)(cfg.models)

        if self.cfg.trainer.val_visualize:
            self.vis_save_dir = os.path.join(self.cfg.save_dir,
                                             'vis_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            mkdir(self.local_rank, self.vis_save_dir)

        # self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    def on_train_start(self) -> None:
        pass
    # def on_train_epoch_start(self) -> None:
    #     if dist.is_available() and dist.is_initialized():
    #         rank = dist.get_rank()
    #     else:
    #         rank = 0

    #     epoch_seed = self.cfg.trainer.seed + self.current_epoch + rank * 10000

    #     seed_everything(epoch_seed, workers=True)
    #     set_seed(epoch_seed)

    #     if self.train_data_loader is not None and hasattr(self.train_data_loader.dataset, self.cfg.trainer.dataset_callback_fn):
    #         getattr(self.train_data_loader.dataset, self.cfg.trainer.dataset_callback_fn)(epoch_seed)
    def training_step(self, batch, batch_idx):
        pred, losses = self.model.forward_train(batch)
        if self.global_step % self.cfg.trainer.log.interval == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.trainer.total_epochs,
                self.global_step,
                batch_idx,
                # self.num_training_batches,
                self.trainer.num_training_batches,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in losses:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, losses[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    losses[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

        return losses['total'].mean()

    def training_step_end(self, batch_parts):
        pass

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.logger.dump_cfg_with_dir(self.cfg, self.cfg.save_dir)
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))
        self.trainer.save_checkpoint(os.path.join(self.logger.log_dir, "model_last.ckpt"))
        self.trainer.save_checkpoint(os.path.join(self.logger.log_dir, "model_" + str(self.current_epoch) + ".ckpt"))
        self.lr_scheduler.step()
        if self.train_data_loader != None:
            seed_everything(self.cfg.trainer.seed + self.current_epoch, workers=True)
            set_seed(self.cfg.trainer.seed + self.current_epoch)
            getattr(self.train_data_loader.dataset, self.cfg.trainer.dataset_callback_fn)(
                    self.cfg.trainer.seed + self.current_epoch)

    def visualize(self, batch, pred):
        import cv2
        # batch_size = batch['image'].shape[0]
        # optimizing_result_imgs = pred['optimizing_result_imgs']

        if 'seg_imgs' in pred:
            batch_size = batch['image'].shape[0]
            for i in range(batch_size):
                output_name = batch['output_name'][i]
                output_seg_path = os.path.join(self.vis_save_dir, output_name+'_seg.png')
                seg_img = pred['seg_imgs'][i]
                cv2.imwrite(output_seg_path, seg_img)
                # weight_img = pred['weight_imgs'][i]
                # output_weight_path = os.path.join(self.vis_save_dir, output_name + '_weight.png')
                # cv2.imwrite(output_weight_path, weight_img)

        if 'd_init_images' in pred:
            for i in range(len(pred['d_init_images'])):
                output_name = batch['output_name'][i]
                cv2.imwrite(os.path.join(self.vis_save_dir, output_name + '_init.png'), pred['d_init_images'][i])
                cv2.imwrite(os.path.join(self.vis_save_dir, output_name + '_opt.png'), pred['d_opt_images'][i])
                cv2.imwrite(os.path.join(self.vis_save_dir, output_name + '_ref.png'), pred['d_ref_images'][i])

        if 'optimizing_result_imgs' in pred:
            for i in range(len(pred['optimizing_result_imgs'])):
                for j, optimizing_result_img in enumerate(pred['optimizing_result_imgs'][i]):
                    output_name = batch['output_name'][j]
                    output_optimizing_result_path = os.path.join(self.vis_save_dir, output_name + '_' + str(i) + '.png')
                    cv2.imwrite(output_optimizing_result_path, optimizing_result_img)

        if 'weight_imgs' in pred:
            for i in range(len(pred['weight_imgs'])):
                for j, weight_img in enumerate(pred['weight_imgs'][i]):
                    output_name = batch['output_name'][j]
                    output_weight_result_path = os.path.join(self.vis_save_dir, output_name + '_' + str(i) + '_weight.png')
                    cv2.imwrite(output_weight_result_path, weight_img)

        if 'optimizing_result_images' in pred:
            for ti, opt_reslut_imgs in enumerate(pred['optimizing_result_images']):
                for b, opt_reslut_img in enumerate(opt_reslut_imgs):
                    output_name = pred['output_names'][ti][b]
                    output_path = os.path.join(self.vis_save_dir, output_name + '.png')
                    cv2.imwrite(output_path, opt_reslut_img)

        if 'box_images' in pred:
            for i in range(len(pred['box_images'])):
                for j, box_image in enumerate(pred['box_images'][i]):
                    output_name = batch['output_names'][i+1][j]
                    output_box_image_path = os.path.join(self.vis_save_dir, output_name + '_' + str(i+1) + '_box.png')
                    cv2.imwrite(output_box_image_path, box_image)

        if 'prob_optimizing_result_images' in pred:
            for ti, opt_reslut_imgs in enumerate(pred['prob_optimizing_result_images']):
                for b, opt_reslut_img in enumerate(opt_reslut_imgs):
                    output_name = pred['output_names'][ti+1][b]
                    output_path = os.path.join(self.vis_save_dir, output_name + '_prob.png')
                    cv2.imwrite(output_path, opt_reslut_img)

        if 'refine_optimizing_result_images' in pred:
            for ti, opt_reslut_imgs in enumerate(pred['refine_optimizing_result_images']):
                for b, opt_reslut_img in enumerate(opt_reslut_imgs):
                    output_name = pred['output_names'][ti+1][b]
                    output_path = os.path.join(self.vis_save_dir, output_name + '_refine.png')
                    cv2.imwrite(output_path, opt_reslut_img)
        
        if 'reference_images' in pred:
            for ti, opt_reference_images in enumerate(pred['reference_images']):
                for i, reference_images in enumerate(opt_reference_images):
                    for b, reference_image in enumerate(reference_images):
                        output_name = pred['output_names'][ti][b]
                        output_path = os.path.join(self.vis_save_dir, output_name + f'_ref_{i}.png')
                        cv2.imwrite(output_path, reference_image)

        if 'uncertainties_render_maps' in pred:
            for ti, uncertainties_render_maps in enumerate(pred['uncertainties_render_maps']):
                for s, uncertainties_render_map in enumerate(uncertainties_render_maps):
                    for b, uncertainty in enumerate(uncertainties_render_map):
                        output_name = pred['output_names'][ti][b]
                        output_path = os.path.join(self.vis_save_dir, output_name + f'_ucr_{s}.png')
                        cv2.imwrite(output_path, uncertainty)
        
        if 'uncertainties_maps' in pred:
            for ti, uncertainties_maps in enumerate(pred['uncertainties_maps']):
                for s, uncertainties_map in enumerate(uncertainties_maps):
                    for b, uncertainty in enumerate(uncertainties_map):
                        output_name = pred['output_names'][ti+1][b]
                        output_path = os.path.join(self.vis_save_dir, output_name + f'_ucq_{s}.png')
                        cv2.imwrite(output_path, uncertainty)

        if 'uncertainties_p' in pred:
            for ti, uncertainties_maps in enumerate(pred['uncertainties_p']):
                for s, uncertainties_map in enumerate(uncertainties_maps):
                    for b, uncertainty in enumerate(uncertainties_map):
                        output_name = pred['output_names'][ti+1][b]
                        output_path = os.path.join(self.vis_save_dir, output_name + f'_p_{s}.png')
                        cv2.imwrite(output_path, uncertainty)
        
        if 'uncertainties_c' in pred:
            for ti, uncertainties_maps in enumerate(pred['uncertainties_c']):
                for s, uncertainties_map in enumerate(uncertainties_maps):
                    for b, uncertainty in enumerate(uncertainties_map):
                        output_name = pred['output_names'][ti][b]
                        output_path = os.path.join(self.vis_save_dir, output_name + f'_c_{s}.png')
                        cv2.imwrite(output_path, uncertainty)

        if 'pred_result_images' in pred:
            for i, opt_reslut_imgs in enumerate(pred['pred_result_images']):
                for b, opt_reslut_img in enumerate(opt_reslut_imgs):
                    ti = i + 1
                    output_name = pred['output_names'][ti][b]
                    output_path = os.path.join(self.vis_save_dir, output_name + '_pred.png')
                    cv2.imwrite(output_path, opt_reslut_img)

        # import ipdb
        # ipdb.set_trace()

    def validation_step(self, batch, batch_idx):
        pred, losses, metrics = self.model.forward_eval(batch, self.cfg.trainer.val_visualize)
        
        if batch_idx % self.cfg.trainer.log.interval == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.trainer.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            for loss_name in losses:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, losses[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        if self.cfg.trainer.val_visualize:
            self.visualize(batch, pred)

        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        results = {}
        for res in validation_step_outputs:
            for key, value in res.items():
                if key not in results:
                    results[key] = value
                else:
                    results[key] = torch.cat((results[key], value))

        if dist.is_available() and dist.is_initialized():
            all_results = gather_results(results)
        else:
            all_results = results

        if all_results:
            metrics = all_results
            for key in metrics.keys():
                metrics[key] = metrics[key].float().mean().item()
            self.logger.log_metrics(metrics, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))

    def test_step(self, batch, batch_idx):
        pred, losses, metrics = self.model.forward_eval(batch, self.cfg.trainer.val_visualize)

        if batch_idx % self.cfg.trainer.log.interval == 0:
            # lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Test|Iter{}({})| ".format(
                # self.current_epoch + 1,
                # self.cfg.trainer.total_epochs,
                self.global_step,
                batch_idx,
                # lr,
            )
            for loss_name in losses:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, losses[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        if self.cfg.trainer.val_visualize:
            self.visualize(batch, pred)

        return metrics

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            for key, value in res.items():
                if key not in results:
                    results[key] = value
                else:
                    results[key] = torch.cat((results[key], value))

        if dist.is_available() and dist.is_initialized():
            all_results = gather_results(results)
        else:
            all_results = results

        if all_results:
            
            metrics = {'5cm_5d': ((['R_error'] < 5) & (all_results['t_error'] < 0.05)).float().mean().item(),
                       '2cm_2d': ((all_results['R_error'] < 2) & (all_results['t_error'] < 0.02)).float().mean().item(),
                       'ADD_0.1d': (all_results['err_add'] < all_results['diameter'] * 0.1).float().mean().item(),
                       'ADD_0.05d': (all_results['err_add'] < all_results['diameter'] * 0.05).float().mean().item(),
                       'ADD_0.02d': (all_results['err_add'] < all_results['diameter'] * 0.02).float().mean().item(),
                       'ADD_S_0.1d': (all_results['err_add_s'] < all_results['diameter'] * 0.1).float().mean().item(),
                       'ADD_S_0.05d': (all_results['err_add_s'] < all_results['diameter'] * 0.05).float().mean().item(),
                       'ADD_S_0.02d': (all_results['err_add_s'] < all_results['diameter'] * 0.02).float().mean().item(),
                       'ADD(S)_0.1d': (all_results['err_add(s)'] < all_results['diameter'] * 0.1).float().mean().item(),
                       'ADD(S)_0.05d': (all_results['err_add(s)'] < all_results['diameter'] * 0.05).float().mean().item(),
                       'ADD(S)_0.02d': (all_results['err_add(s)'] < all_results['diameter'] * 0.02).float().mean().item(),
                       'ADD(S)_0.1d_init': (all_results['err_add(s)_init'] < all_results['diameter'] * 0.1).float().mean().item(),
                       'ADD(S)_0.05d_init': (all_results['err_add(s)_init'] < all_results['diameter'] * 0.05).float().mean().item(),
                       'ADD(S)_0.02d_init': (all_results['err_add(s)_init'] < all_results['diameter'] * 0.02).float().mean().item()}

            self.logger.log_metrics(metrics, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))
    
    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.cfg.trainer.optimizer)
        build_optimizer = getattr(torch.optim, optimizer_cfg.name)
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        lr_params = pack_lr_parameters(
            params, self.cfg.trainer.optimizer.lr, self.cfg.trainer.optimizer.lr_scaling, self.logger)
        # optimizer = build_optimizer(self.parameters(), lr=self.cfg.trainer.optimizer.lr)
        optimizer = build_optimizer(lr_params, lr=self.cfg.trainer.optimizer.lr)

        schedule_cfg = copy.deepcopy(self.cfg.trainer.lr_schedule)
        schedule_cfg = OmegaConf.to_container(schedule_cfg, resolve=True)
        build_scheduler = getattr(torch.optim.lr_scheduler, schedule_cfg.pop('name'))
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)

        return optimizer
    
    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

