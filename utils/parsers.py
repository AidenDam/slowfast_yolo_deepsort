"""Argument parser functions."""

import sys
import argparse

from config import get_cfg, assert_and_infer_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--input', type=str, default="0", help='test imgs folder or video or camera')
    parser.add_argument('--output_dir', type=str, default="output/", help='folder to save result')
    parser.add_argument('--output_video', type=str, default="output.mp4", help='file name to save result video')
    parser.add_argument('--save_all_video', type=bool, default=False, help='whether save all video or only detected frame')
    parser.add_argument('--output_csv', type=str, default="output.csv", help='file name to save result csv')
    # object detect config
    parser.add_argument('--yolo_ckpt', default='weights/yolov8m.onnx', help='yolo checkpoint')
    # tracker config
    parser.add_argument('--tracker_ckpt', default='weights/ckpt.t7', help='deepsort checkpoint')
    # SlowFast config
    parser.add_argument('--ava_action_list', default='config/ava_action_list.pbtxt', help='ava action list')
    parser.add_argument('--slowfast_ckpt', default='weights/SLOWFAST_32x2_R50_DETECTION.pyth', help='slowfast checkpoint')
    parser.add_argument(
        "--slowfast_cfg",
        dest="slowfast_cfg_file",
        help="Path to the SlowFast config file",
        default="config/SLOWFAST.yaml",
        type=str
    )
    # More config
    parser.add_argument(
        "--opts",
        help="See config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    assert_and_infer_cfg(cfg)

    return cfg
