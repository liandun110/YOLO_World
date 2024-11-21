# import os
# os.system("pip uninstall -y mmcv-full")
# os.system("mim install 'mmengine>=0.6.0'")
# # os.system("pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html")
# os.system("mim install 'mmcv-lite==2.0.1'")
# os.system("mim install 'mmdet>=3.0.0,<4.0.0'")
# os.system("mim install 'mmyolo'")
# os.system("pip install -e .")
#
# import argparse
# import os.path as osp
#
# from mmengine.config import Config, DictAction
# from mmengine.runner import Runner
# from mmengine.dataset import Compose
# from mmyolo.registry import RUNNERS
#
# from tools.demo import demo
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='YOLO-World Demo')
#     parser.add_argument('--config', default='configs/pretrain/yolo_world_xl_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
#     parser.add_argument('--checkpoint', default='yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth')
#     parser.add_argument(
#         '--work-dir',
#         help='the directory to save the file containing evaluation metrics')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#     args = parse_args()
#
#     # load config
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#
#     if args.work_dir is not None:
#         cfg.work_dir = args.work_dir
#     elif cfg.get('work_dir', None) is None:
#         cfg.work_dir = osp.join('./work_dirs',
#                                 osp.splitext(osp.basename(args.config))[0])
#
#     cfg.load_from = args.checkpoint
#
#     if 'runner_type' not in cfg:
#         runner = Runner.from_cfg(cfg)
#     else:
#         runner = RUNNERS.build(cfg)
#
#     runner.call_hook('before_run')
#     runner.load_or_resume()
#     pipeline = cfg.test_dataloader.dataset.pipeline
#     runner.pipeline = Compose(pipeline)
#     runner.model.eval()
#     demo(runner, args, cfg)
import argparse
import os
import os.path as osp
import cv2
import torch
import numpy as np
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS
from tools.demo import run_image  # 导入 run_image 函数


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('text', help='Comma separated list of classes to detect')
    parser.add_argument('--config',
                        default='configs/pretrain/yolo_world_xl_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
    parser.add_argument('--checkpoint', default='yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth')
    parser.add_argument('--output', default='./output.png', help='Path to save the output image')
    parser.add_argument('--work-dir', help='Directory to save the evaluation metrics')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override settings in the config')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # Initialize the model
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    # Run image detection
    image = Image.open(args.image_path)  # Open image from file path
    output_image = run_image(runner, image, args.text, max_num_boxes=100, score_thr=0.05, nms_thr=0.5,
                             image_path=args.output)

    # Save the output image
    output_image.save(args.output)
    print(f"Output saved to {args.output}")


if __name__ == '__main__':
    main()
