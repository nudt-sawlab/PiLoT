from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

# from mylib.pytorch_lightning.base_module import load_pretrained_dict
from .base_model import BaseModel
# from .mobileone import MobileOneBlock
# import segmentation_models_pytorch as smp

from segmentation_models_pytorch.encoders import get_encoder, get_preprocessing_fn, get_preprocessing_params
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock, CenterBlock
# from segmentation_models_pytorch.decoders.fpn.decoder

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        xs = []
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            xs.append(x)

        return xs

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BackboneUnet(BaseModel):
    default_conf = {
        'num_output_layer': 3,
        'output_dim': [16, 16, 16],
        'encoder': 'mobileone_s0',
        'encoder_depth': 5,
        'decoder': 'UnetDecoder',
        'decoder_channels': [256, 128, 64, 32, 16],
        'align_data_to_pretrain': False,
        'pretrained_weights': 'imagenet',
        'train_mask': False,
        'normalize_key': False,
        'train_box': False,
        'compute_uncertainty': False
    }

    required_data_keys = {

    }

    def _init(self, conf):
        # self.test_unet = smp.Unet(encoder_name='mobileone_s0', 
        #                          encoder_weights='imagenet',
        #                          in_channels=3, classes=16)
        self.conf = conf
        encoder_depth = conf.encoder_depth  #3
        decoder_channels = conf.decoder_channels  # [128, 64, 32]
        decoder_use_batchnorm = True
        decoder_attention_type = None
        pretrained_weights = conf.pretrained_weights # 'imagenet'

        if conf.align_data_to_pretrain:  #False
            self.preprocess_params = get_preprocessing_params(conf.encoder, pretrained=pretrained_weights)
            # self.preprocess_input = get_preprocessing_fn(conf.encoder, pretrained=pretrained_weights)
            self.mean = torch.from_numpy(np.asarray(self.preprocess_params['mean'])).float()
            self.std = torch.from_numpy(np.asarray(self.preprocess_params['std'])).float()
        else:
            self.preprocess_params = None
            # self.preprocess_input = None

        self.encoder = get_encoder(
            conf.encoder,  # mobileone_s0
            in_channels=3,
            depth=encoder_depth,
            weights=pretrained_weights,
        )
        
        decoder_class = getattr(sys.modules[__name__], conf.decoder)
        self.decoder = decoder_class(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.smooth_layers = nn.ModuleList()
        for i, output_dim in enumerate(conf.output_dim):
            idx = conf.num_output_layer - i  #3
            self.smooth_layers.append(nn.Conv2d(in_channels=decoder_channels[-idx], out_channels=output_dim,
                                                kernel_size=3, stride=1, padding=1))

        if conf.compute_uncertainty:  #True
            self.uncertainty_layers = nn.ModuleList()
            for i, output_dim in enumerate(conf.output_dim):  #[32,32,32]
                idx = conf.num_output_layer - i  #3
                self.uncertainty_layers.append(nn.Conv2d(in_channels=decoder_channels[-idx], out_channels=1,
                                                        kernel_size=1, stride=1, padding=0))

        if conf.train_mask:  #False
            self.key_proj = nn.Conv2d(self.encoder.out_channels[-1], 64, kernel_size=3, padding=1)
            # self.d_proj = nn.Conv2d(self.encoder.out_channels[-1], 1, kernel_size=3, padding=1)
            # self.e_proj = nn.Conv2d(self.encoder.out_channels[-1], 64, kernel_size=3, padding=1)
            
            self.value_encoder = nn.Conv2d(self.encoder.out_channels[-1]+1, self.encoder.out_channels[-1], kernel_size=3, padding=1)

            # self.segment_encoder = get_encoder(
            #     conf.encoder,
            #     in_channels=4,
            #     depth=encoder_depth,
            #     # weights=pretrained_weights,
            # )
            # self.value_encoder = nn.Sequential(
            #     nn.Conv2d(self.encoder.out_channels[-1]*2, self.encoder.out_channels[-1], kernel_size=3, padding=1),
            #     nn.ReLU(),
            #     nn.Conv2d(self.encoder.out_channels[-1], self.encoder.out_channels[-1], kernel_size=3, padding=1),
            #     nn.ReLU()
            # )

            self.x1conv = nn.Sequential(
                nn.Conv2d(self.encoder.out_channels[-1]*2, self.encoder.out_channels[-1], 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.encoder.out_channels[-1], self.encoder.out_channels[-1], 3, padding=1),
                nn.ReLU()
            )
            if conf.train_box == False:  #False
                self.segment_decoder = decoder_class(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=encoder_depth,
                    use_batchnorm=decoder_use_batchnorm,
                    center=False,
                    attention_type=decoder_attention_type,
                )
                self.mask_layers = nn.ModuleList()
                for i, output_dim in enumerate(conf.output_dim):
                    idx = conf.num_output_layer - i
                    self.mask_layers.append(nn.Conv2d(in_channels=decoder_channels[-idx], out_channels=1,
                                                    kernel_size=1, stride=1, padding=0))
            else:
                self.segment_encoder = get_encoder(
                    self.conf.encoder,
                    in_channels=self.encoder.out_channels[-1],
                    depth=5,
                    # weights=pretrained_weights,
                )
                self.box_mlp = MLP(1024, 512, 4, 3)

    def _forward_encode_key_and_value(self, x, mask=None):
        if self.conf.align_data_to_pretrain:  #False
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x = x - mean[None, :, None, None]
            x = x / std[None, :, None, None]
        x1 = self.encoder(x)
        x2 = self.decoder(*x1)

        output_layers = x2[-self.conf.num_output_layer:]
        outputs = []
        for smooth_layer, output_layer in zip(self.smooth_layers, output_layers):
            outputs.append(smooth_layer(output_layer))
        
        keys = self.key_proj(x1[-1])
        if self.conf.normalize_key:  #False
            keys = torch.nn.functional.normalize(keys, dim=1)
        # shrinkage = self.d_proj(x1[-1])**2 + 1
        # selection = torch.sigmoid(self.e_proj(x1[-1]))
        if mask is not None:
            mask_scale = torch.nn.functional.interpolate(mask, size=x1[-1].shape[-2:], mode='nearest')
            inp = torch.cat((x1[-1], mask_scale), dim=1)
            values = self.value_encoder(inp)
            return outputs, x1, keys, values
            # return outputs, x1, keys, shrinkage, selection, values
        else:
            return outputs, x1, keys
            # return outputs, x1, keys, shrinkage, selection
    
    def _forward_with_encode_key(self, x):
        if self.conf.align_data_to_pretrain:  #False
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x = x - mean[None, :, None, None]
            x = x / std[None, :, None, None]
        x1 = self.encoder(x)
        x2 = self.decoder(*x1)

        output_layers = x2[-self.conf.num_output_layer:]
        outputs = []
        for smooth_layer, output_layer in zip(self.smooth_layers, output_layers):
            outputs.append(smooth_layer(output_layer))
        
        keys = self.key_proj(x1[-1])
        return outputs, x1, keys
        
    def _forward_encode_value(self, x, mask, x1):
        inp = torch.cat((x, mask), dim=1)
        x3 = self.segment_encoder(inp)
        inp = torch.cat((x3[-1], x1), dim=1)
        values = self.value_encoder(inp)
        return values

    def _forward_segment(self, x1, readout):
        readout = readout.view(*x1[-1].shape)
        x = torch.cat((x1[-1], readout), dim=1)
        x = self.x1conv(x)
        if self.conf.train_box == False:  #False
            x1new = x1[:-1]
            x1new.append(x)
            x2 = self.segment_decoder(*x1new)
            output_layers = x2[-self.conf.num_output_layer:]
            outputs = []
            for mask_layer, output_layer in zip(self.mask_layers, output_layers):
                outputs.append(torch.sigmoid(mask_layer(output_layer)))

            return outputs
        else:
            x2 = self.segment_encoder(x)
            x3 = x2[-1][..., 0, 0]
            x4 = self.box_mlp(x3)
            box = torch.sigmoid(x4)
            
            return box

    def _forward(self, x):
        # x0 = self.preprocess_input(x)
        # x = self.preprocess_input(x) if self.conf.align_data_to_pretrain else x
        if self.conf.align_data_to_pretrain:  #False
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x = x - mean[None, :, None, None]
            x = x / std[None, :, None, None]
        image = x['image']
        x1 = self.encoder(image)
        x2 = self.decoder(*x1)

        output_layers = x2[-self.conf.num_output_layer:]
        outputs = []
        for smooth_layer, output_layer in zip(self.smooth_layers, output_layers):
            outputs.append(smooth_layer(output_layer))
        pred = {'feature_maps': list(reversed(outputs))}
        uncertainties = []
        if self.conf.compute_uncertainty:
            for uncertainty_layer, output_layer in zip(self.uncertainty_layers, output_layers):
                uncertainties.append(torch.sigmoid(uncertainty_layer(output_layer)))
            pred['confidences'] = list(reversed(uncertainties))
        return pred

    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def metrics(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError