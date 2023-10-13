# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torchvision.ops.roi_align as roi_align
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch.utils.data
import torch.utils.data.distributed

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.efficientnet import EfficientFace, efficient_face
from slowfast.models.recorder import RecorderMeter
from slowfast.models.modulator import Modulator

from slowfast.models.resnet import ResNet, resnet50
from slowfast.models.stem_helper import PatchEmbed
from slowfast.models.utils import (
    round_width,
    validate_checkpoint_wrapper_import,
)

from slowfast.models.vit import TimeSformer
from torchvision import transforms

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

# Facial Expression Based PD Classification Model
@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

        self.st_func = {
            "temporal_attention_region_attention": self.temporal_attention_region_attention
        }

        ########### Config Parameters ###########
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.use_max_pool = True
        self.use_s5 = False
        self.roi_align_size = (3, 3)
        self.use_softmax = True
        self.vit_embed_dim = 768
        self.use_vit = False
        self.use_mlp = False
        self.use_bn = True
        self.st_config = ["temporal_attention", "region_attention"]
        self.use_pos_embed = True
        self.use_time_embed = True
        self.image_variant = "video+region"
        self.use_imagenet_resnet = True
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.feature_size = 768
        self.backbone = "vgg"
        ########################################

        self.cls_head = None
        self.vit_proj = None
        self.vit = None
        self.temporal_pool = None
        self.region_pool = None
        self.temporal_attention = None
        self.region_attention = None
        self.pos_embed = None
        self.pos_drop = None
        self.time_embed = None
        self.time_drop = None
        self.image_mlp = None
        self.resnet = None
        self.img_preprocess = None
        
        self.proj = nn.Linear(24576, 768)
        self.reg_proj = nn.Linear(9216, 768)
        self.att_proj = nn.Linear(1440, 768)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.resnet = torch.nn.DataParallel(self.resnet).cuda()
        backbone = torch.load(cfg.DATA.BACKBONE_PATH)
        for key in list(backbone['state_dict']):
            newKeyName = key.replace(".base_net", "")
            backbone['state_dict'][newKeyName] = backbone['state_dict'].pop(key)
        
        backbone['state_dict']["module.projection_net.net.0.weight"] = torch.rand(1000, 2048)
        backbone['state_dict']['module.projection_net.net.0.bias'] = torch.rand(1000)
        backbone['state_dict']['module.fc.weight'] = backbone['state_dict']["module.projection_net.net.0.weight"]
        backbone['state_dict']['module.fc.bias'] = backbone['state_dict']['module.projection_net.net.0.bias']
        backbone['state_dict'] = {k: v for k, v in backbone['state_dict'].items() if '.projection_net' not in k}
        backbone['state_dict'] = {k: v for k, v in backbone['state_dict'].items() if '.prototypes' not in k}
        self.resnet.load_state_dict(backbone['state_dict'])

        self.img_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.reg_mlp1 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp2 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp3 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp4 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp5 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp6 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp7 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp8 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp9 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp10 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp11 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp12 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp13 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))
        self.reg_mlp14 = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 2))

        self.image_mlp = nn.Sequential(nn.Linear(self.feature_size, 384), nn.ReLU(), nn.Linear(384, self.num_classes))

        if self.use_mlp:
            self.cls_head = nn.Sequential(nn.Linear(768, 4608), nn.ReLU(), nn.Linear(4608, 384), nn.ReLU(), nn.Linear(384, self.num_classes))
        else:
            self.cls_head = None

        if self.use_vit:
            self.vit = TimeSformer(img_size=256, num_classes=2, num_frames=8, attention_type='divided_space_time', depth=1, embed_dim=1536, pretrained_model='')

        if self.use_max_pool:
            self.temporal_pool = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))
            self.region_pool = nn.MaxPool2d((1, 2), stride=(1, 2))
        else:
            self.temporal_pool = nn.AvgPool3d((1, 1, 2), stride=(1, 1, 2))
            self.region_pool = nn.AvgPool2d((1, 2), stride=(1, 2))
        
        if self.st_config is not None:
            self.temporal_attention = Attention(self.feature_size, num_heads=3) if "temporal_attention" in self.st_config else None
            self.region_attention = Attention(self.feature_size, num_heads=3) if "region_attention" in self.st_config else None
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_size))

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 14+1, self.feature_size))
        
        if self.use_time_embed:
            if "temporal_attention" in self.st_config and "region_attention" in self.st_config:
                self.time_embed = nn.Parameter(torch.zeros(1, 8+1, self.feature_size))
            else:
                self.time_embed = nn.Parameter(torch.zeros(1, 8+1, self.feature_size))

        self.act = nn.Softmax(dim=1)
        self.out_batchnorm = nn.BatchNorm1d(6)
        self.reg_batchnorm = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()
        self.video_time_embed = nn.Parameter(torch.zeros(1, 8, self.feature_size))

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.cfg = cfg

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

    def temporal_attention_region_attention(self, x):
        batch_size, num_frames, num_regions, region_fv = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape((batch_size*num_frames, num_regions, region_fv))

        if self.pos_embed is not None:
            x = x + self.pos_embed

        out = self.region_attention(x)

        x = x.reshape((batch_size*num_regions, num_frames, region_fv))

        cls_tokens = self.cls_token.expand(batch_size*num_regions, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.time_embed is not None:
            x = x + self.time_embed

        out = self.temporal_attention(x)
        final_cls = out[:, 0]
        video_cls = final_cls.clone()

        video_cls = video_cls.reshape((batch_size, num_frames, -1))
        video_cls = torch.mean(video_cls, 1, True)
        video_cls = torch.squeeze(video_cls)
        video_cls = self.att_proj(video_cls)

        final_cls = final_cls.reshape((batch_size, num_regions, -1))
        final_cls[:, 0] = video_cls
        print(final_cls.shape)

        return final_cls
        
    def forward(self, x, bboxes): 
        print("RUNNING", self.image_variant)
        x = x[:]  # avoid pass by reference
        batch_size = x[0].shape[0]
        if self.use_imagenet_resnet:
            x = torch.squeeze(x[0])
            print(x.shape)
            a, b, c, d, e = x.shape
            x = torch.chunk(x, c, dim=2)
           
            y = []
            for i in x:
                x = torch.squeeze(i)
                x = self.img_preprocess(x)
                
                x = self.resnet.module.conv1(x)
                x = self.resnet.module.bn1(x)
                x = self.resnet.module.maxpool(x)
                x = self.resnet.module.layer1(x)
                x = self.resnet.module.layer2(x)
                x = self.resnet.module.layer3(x)
                x = torch.unsqueeze(x, 2)
                y.append(x)
            # output is of shape [64, 2048, 7, 7]
            x = torch.cat(y, dim=2)
            x = [x]

        fmap_dim = x[0].shape[4]
        batch_size = x[0].shape[0]
        
        # shape of z: (batch_size, num_frames, num_feature_maps, fmap_dim, fmap_dim)
        z_shape = x[0].shape
        z = [x[0].permute((0, 2, 1, 3, 4))]

        # shape of bboxes: (16, 8, 14, 4, 2) ---> (batch_size, frames, regions, bounding box corners, coordinattes)
        batch_size = bboxes.shape[0]
        frames = bboxes.shape[1]
        regions = bboxes.shape[2]
        corners = bboxes.shape[3]
        coords = bboxes.shape[4]

        formatted_bboxes = []
        for elt in range(batch_size):
            for frame in range(frames):
                for region in range(regions):
                    x_min, y_min = bboxes[elt][frame][region][0][0], bboxes[elt][frame][region][0][1]
                    x_max, y_max = bboxes[elt][frame][region][2][0], bboxes[elt][frame][region][2][1]

                    scale_factor = 256 / fmap_dim

                    ### DEBUG ###
                    # assert scale_factor == 36.5714285714
                    #############

                    x_min, y_min = x_min // scale_factor, y_min // scale_factor
                    x_max, y_max = x_max // scale_factor, y_max // scale_factor

                    formatted_bboxes.append([elt*frames + frame, x_min, y_min, x_max, y_max])

        formatted_bboxes = torch.Tensor(formatted_bboxes)
        formatted_bboxes = formatted_bboxes.to(device='cuda')
        batch_size = z[0].shape[0]
        num_frames = z[0].shape[1]
        z = [z[0].reshape((batch_size*num_frames, z[0].shape[2], z[0].shape[3], z[0].shape[4]))]
        feature_maps = roi_align(z[0], formatted_bboxes, self.roi_align_size)
        feature_maps = feature_maps.reshape((batch_size, frames, regions, -1))

        x[0] = x[0].permute(0, 2, 1, 3, 4)
        x_shape = x[0].shape
        video_level_features = x[0].reshape(batch_size, x_shape[1], x_shape[2], x_shape[3]*x_shape[4])
        video_level_features = self.region_pool(video_level_features)
        video_level_features = self.region_pool(video_level_features)
        video_level_features = self.region_pool(video_level_features)
        x_shape = video_level_features.shape
        video_level_features = video_level_features.reshape(batch_size, x_shape[1], 1, x_shape[2]*x_shape[3])
        video_level_features = self.proj(video_level_features)
        feature_maps = self.reg_proj(feature_maps)

        video_level_features = torch.cat((video_level_features, feature_maps), dim=2)
        video_level_features = self.st_func["_".join(self.st_config)](video_level_features)
        feature_shape = video_level_features.shape
        video_level_features = video_level_features.reshape((batch_size, feature_shape[1], -1))
        
        x = torch.chunk(video_level_features, 15, dim=1)
        vid = self.image_mlp(x[0])
        reg1 = self.reg_mlp1(x[1])
        reg2 = self.reg_mlp2(x[2])
        reg3 = self.reg_mlp3(x[3])
        reg4 = self.reg_mlp4(x[4])
        reg5 = self.reg_mlp5(x[5])
        reg6 = self.reg_mlp1(x[6])
        reg7 = self.reg_mlp1(x[7])
        reg8 = self.reg_mlp1(x[8])
        reg9 = self.reg_mlp1(x[9])
        reg10 = self.reg_mlp1(x[10])
        reg11 = self.reg_mlp1(x[11])
        reg12 = self.reg_mlp1(x[12])
        reg13 = self.reg_mlp1(x[13])
        reg14 = self.reg_mlp1(x[14])
        out = torch.stack((reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14), dim=1)
        out = torch.squeeze(out)
        out_shape = out.shape
        out = out.reshape(out_shape[0], out_shape[2], out_shape[1])
        out = self.reg_batchnorm(out)
        out = out.reshape(out_shape)
        out = self.act(out)

        vid = torch.squeeze(vid)
        vid = self.out_batchnorm(vid)
        vid = self.act(vid)
        return vid, out, None