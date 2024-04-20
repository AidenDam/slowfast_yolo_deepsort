from functools import partial

import torch
import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.models.resnet import create_bottleneck_block
from pytorchvideo.models.head import create_res_roi_pooling_head

from .roi_align import ROIAlign
from .helper_function import get_head_act, get_norm

class SlowFast(nn.Module):
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

        # Params from configs.
        norm_module = get_norm(cfg)
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        spatial_dilations = cfg.RESNET.SPATIAL_DILATIONS
        spatial_strides = cfg.RESNET.SPATIAL_STRIDES
        temp_kernel = [
            [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
        ]
        num_block_temp_kernel = cfg.RESNET.NUM_BLOCK_TEMP_KERNEL
        stage_depth = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        stage_conv_a_kernel_sizes = [[], []]
        for pathway in range(2):
            for stage in range(4):
                stage_conv_a_kernel_sizes[pathway].append(
                    ((temp_kernel[stage + 1][pathway][0], 1, 1),)
                    * num_block_temp_kernel[stage][pathway]
                    + ((1, 1, 1),)
                    * (
                        stage_depth[stage]
                        - num_block_temp_kernel[stage][pathway]
                    )
                )
        
        # Head from config
        # Number of stages = 4
        stage_dim_in = cfg.RESNET.WIDTH_PER_GROUP * 2 ** (4 + 1)
        head_in_features = stage_dim_in + stage_dim_in // cfg.SLOWFAST.BETA_INV
        
        self.detection_head = create_res_roi_pooling_head(
                in_features=head_in_features,
                out_features=cfg.MODEL.NUM_CLASSES,
                pool=None,
                output_size=(1, 1, 1),
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                activation=None,
                output_with_global_average=False,
                pool_spatial=nn.MaxPool2d,
                resolution=[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2,
                spatial_scale=1.0 / float(cfg.DETECTION.SPATIAL_SCALE_FACTOR),
                sampling_ratio=0,
                roi=ROIAlign,
            )
        head_pool_kernel_sizes = (
            (
                cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA,
                1,
                1,
            ),
            (cfg.DATA.NUM_FRAMES, 1, 1),
        )

        self.model = create_slowfast(
            # SlowFast configs.
            slowfast_channel_reduction_ratio=cfg.SLOWFAST.BETA_INV,
            slowfast_conv_channel_fusion_ratio=cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            slowfast_fusion_conv_kernel_size=(
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                1,
                1,
            ),
            slowfast_fusion_conv_stride=(cfg.SLOWFAST.ALPHA, 1, 1),
            # Input clip configs.
            input_channels=cfg.DATA.INPUT_CHANNEL_NUM,
            # Model configs.
            model_depth=cfg.RESNET.DEPTH,
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            # Normalization configs.
            norm=norm_module,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_outs=(
                width_per_group,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ),
            stem_conv_kernel_sizes=(
                (temp_kernel[0][0][0], 7, 7),
                (temp_kernel[0][1][0], 7, 7),
            ),
            stem_conv_strides=((1, 2, 2), (1, 2, 2)),
            stem_pool=nn.MaxPool3d,
            stem_pool_kernel_sizes=((1, 3, 3), (1, 3, 3)),
            stem_pool_strides=((1, 2, 2), (1, 2, 2)),
            # Stage configs.
            stage_conv_a_kernel_sizes=stage_conv_a_kernel_sizes,
            stage_conv_b_kernel_sizes=(
                ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
                ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
            ),
            stage_conv_b_num_groups=(
                (num_groups, num_groups, num_groups, num_groups),
                (num_groups, num_groups, num_groups, num_groups),
            ),
            stage_conv_b_dilations=(
                (
                    (1, spatial_dilations[0][0], spatial_dilations[0][0]),
                    (1, spatial_dilations[1][0], spatial_dilations[1][0]),
                    (1, spatial_dilations[2][0], spatial_dilations[2][0]),
                    (1, spatial_dilations[3][0], spatial_dilations[3][0]),
                ),
                (
                    (1, spatial_dilations[0][1], spatial_dilations[0][1]),
                    (1, spatial_dilations[1][1], spatial_dilations[1][1]),
                    (1, spatial_dilations[1][1], spatial_dilations[1][1]),
                    (1, spatial_dilations[1][1], spatial_dilations[1][1]),
                ),
            ),
            stage_spatial_strides=(
                (
                    spatial_strides[0][0],
                    spatial_strides[1][0],
                    spatial_strides[2][0],
                    spatial_strides[3][0],
                ),
                (
                    spatial_strides[0][1],
                    spatial_strides[1][1],
                    spatial_strides[2][1],
                    spatial_strides[3][1],
                ),
            ),
            stage_temporal_strides=((1, 1, 1, 1), (1, 1, 1, 1)),
            bottleneck=create_bottleneck_block,
            # Head configs.
            head=None,
            head_pool=nn.AvgPool3d,
            head_pool_kernel_sizes=head_pool_kernel_sizes,
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x, bboxes):
        if bboxes.shape[0] == 0:
            return torch.tensor([])

        x = self.model(x)
        x = self.detection_head(x, bboxes)
        x = self.post_act(x)

        x = x.view(x.shape[0], -1)
        return x
    
    def load_checkpoint(self, checkpoint_path: str, strict=True, map_location=None):
        print(f'Loading weights from {checkpoint_path}...', end=' ')
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint["model_state"]
        self.load_state_dict(state_dict, strict=strict)
        print('Done!')

        return self
