from fvcore.common.config import CfgNode

_C = CfgNode(new_allowed=True)

# ---------------------------------------------------------------------------- #
# YOLO
# ---------------------------------------------------------------------------- #
_C.YOLO = CfgNode(new_allowed=True)

# inference size (pixels)
_C.YOLO.IMSIZE = 640

# object confidence threshold
_C.YOLO.CONF = 0.5

# IOU threshold for NMS
_C.YOLO.IOU = 0.5


# ---------------------------------------------------------------------------- #
# DEEPSORT
# ---------------------------------------------------------------------------- #
_C.DEEPSORT = CfgNode(new_allowed=True)

_C.DEEPSORT.MAX_DIST = 0.2

_C.DEEPSORT.MIN_CONFIDENCE = 0.3

_C.DEEPSORT.NMS_MAX_OVERLAP = 0.4

_C.DEEPSORT.MAX_IOU_DISTANCE = 0.7

_C.DEEPSORT.MAX_AGE = 70

_C.DEEPSORT.N_INIT = 3

_C.DEEPSORT.NN_BUDGET = 100


# ---------------------------------------------------------------------------- #
# SLOWFAST
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode(new_allowed=True)

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 1

# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False


# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode(new_allowed=True)

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3, 3], [4, 4], [6, 6], [3, 3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1, 1], [1, 1], [1, 1], [2, 2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1, 1], [2, 2], [2, 2], [1, 1]]


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode(new_allowed=True)

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 4

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 7


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode(new_allowed=True)

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode(new_allowed=True)

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 6

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode(new_allowed=True)

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 16

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 2

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

def assert_and_infer_cfg(cfg):
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0
    

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
