import logging
import pickle
from typing import Optional, Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf as oc
from tqdm import tqdm
import torch

from .model3d import Model3D
from .feature_extractor import FeatureExtractor
from .refiners import PoseRefiner, RetrievalRefiner, BaseRefiner

from ..utils.data import Paths
from ..utils.io import parse_image_lists, parse_retrieval, load_hdf5
from ..utils.quaternions import rotmat2qvec
from ..pixlib.utils.experiments import load_checkpoint, load_experiment
from ..pixlib.models import get_model
from ..pixlib.geometry import Camera

logger = logging.getLogger(__name__)
# TODO: despite torch.no_grad in BaseModel, requires_grad flips in ref interp
torch.set_grad_enabled(False)


class Localizer:
    def __init__(self, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        # self.model3d = Model3D(paths.reference_sfm)
        # self.model3d = None #! 1.
        # cameras = parse_image_lists(paths.query_list, with_intrinsics=True)
        # self.queries = {n: c for n, c in cameras}

        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get('conf', {})
        conf_optim = conf.get('optimizer', {})
        if conf.get('experiment'):
            pipeline = load_experiment(
                    conf.experiment,
                    {'extractor': conf_features, 'optimizer': conf_optim})
            pipeline = pipeline.to(device)
            logger.debug(
                'Use full pipeline from experiment %s with config:\n%s',
                conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        elif conf.get('checkpoint'):
            pipeline = load_checkpoint(
                    conf.checkpoint,
                    conf)
            pipeline = pipeline.to(device)
            # logger.debug(
            #     'Use full pipeline from experiment %s with config:\n%s',
            #     conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert 'name' in conf.features
            
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.conf = conf
        self.device = device
        self.optimizer = optimizer
        self.extractor = FeatureExtractor(
            extractor, device, conf.features.get('preprocessing', {}))

    def run_query(self, name: str, camera: Camera):
        raise NotImplementedError




class RenderLocalizer(Localizer):
    def __init__(self, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(conf, device)
        self.refiner = BaseRefiner(
            self.device, self.optimizer, self.extractor, 
            self.conf.refinement)
    def run_query(self, name: str, camera: Camera, ref_camera: Camera, render_frame, query_T = None, render_T = None, Points_3D_ECEF = None, dd = None, gt_pose_dict = None, last_frame_info = {}, query_resize_ratio = 1,image_query=None):
        ret = self.refiner.refine_query_pose(name, camera, ref_camera, render_frame, query_T, render_T, Points_3D_ECEF, dd = dd, gt_pose_dict=gt_pose_dict,last_frame_info = last_frame_info, query_resize_ratio = query_resize_ratio,image_query= image_query)
        return ret

class RetrievalLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        if paths.global_descriptors is not None:
            global_descriptors = load_hdf5(paths.global_descriptors)
        else:
            global_descriptors = None

        self.refiner = RetrievalRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement, global_descriptors=global_descriptors)

        if paths.hloc_logs is not None:
            logger.info('Reading hloc logs...')
            with open(paths.hloc_logs, 'rb') as f:
                self.logs = pickle.load(f)['loc']
            self.retrieval = {q: [self.model3d.dbs[i].name for i in loc['db']]
                              for q, loc in self.logs.items()}
        elif paths.retrieval_pairs is not None:
            self.logs = None
            self.retrieval = parse_retrieval(paths.retrieval_pairs)
        else:
            raise ValueError

    def run_query(self, name: str, camera: Camera, query_T = None, render_T = None, last_frame_info = None):
        #! 3.  #!!!!!!!!!!change
        # dbs = [self.model3d.name2id[r] for r in self.retrieval[name]]  
        dbs = None
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(name, camera, dbs, loc=loc, query_T=query_T, render_T=render_T, last_frame_info = last_frame_info)
        return ret


class PoseLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        self.refiner = PoseRefiner(
            device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement)

        logger.info('Reading hloc logs...')
        with open(paths.hloc_logs, 'rb') as f:
            self.logs = pickle.load(f)['loc']

    def run_query(self, name: str, camera: Camera):
        loc = self.logs[name]
        if loc['PnP_ret']['success']:
            ret = self.refiner.refine(name, camera, loc)
        else:
            ret = {'success': False}
        return ret
