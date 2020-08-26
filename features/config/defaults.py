
from yacs.config import CfgNode as CN

_C = CN()
_C.GPU = [0]
_C.LOG_INTERVAL = 500

# MODEL related params
_C.MODEL = CN()
_C.MODEL.NAME = 'i3d_resnet50_v1_kinetics400'
_C.MODEL.PRETRAINED = True
_C.MODEL.RESUME = ''
_C.MODEL.MODE = ''

# DATASET related params
_C.DATA = CN()
_C.DATA.DATASET = 'MSVD'
_C.DATA.ROOT = 'datasets/MSVD'
_C.DATA.EXT = 'avi'
_C.DATA.SPLIT = 'train'
_C.DATA.SAMPLES_TYPE = 'frames'
_C.DATA.INPUT_SIZE = 224  # the model input size, transformed to this in the transform()
_C.DATA.CLIP_WIDTH = 340  # the clip width as read from video reader
_C.DATA.CLIP_HEIGHT = 256  # the clip height as read from video reader
_C.DATA.CLIP_LENGTH = 32  # the number of frames in a window
_C.DATA.CLIP_STEP = -1  # the step between keyframes (-1 means every realtime second dependent on FPS)
_C.DATA.CLIP_STRIDE = 1  # the stride of frames in window
_C.DATA.NUM_CLASSES = 400
_C.DATA.TEN_CROP = False
_C.DATA.THREE_CROP = False
_C.DATA.NUM_CROP = 1
_C.DATA.AUGMENTATION = 'v1'


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
