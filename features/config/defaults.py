
from yacs.config import CfgNode as CN

_C = CN()
_C.GPU = [0]
_C.LOG_INTERVAL = 10

# MODEL related params
_C.MODEL = CN()
_C.MODEL.NAME = 'i3d_resnet50_v1_kinetics400'
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
_C.MODEL.SLOW_TEMP_STRIDE = 16
_C.MODEL.FAST_TEMP_STRIDE = 2
_C.MODEL.MODE = ''

# DATASET related params
_C.DATA = CN()
_C.DATA.DATASET = 'MSVD'
_C.DATA.ROOT = 'datasets/MSVD'
_C.DATA.EXT = 'avi'
_C.DATA.SPLIT = 'train'
_C.DATA.INPUT_SIZE = 224
_C.DATA.NEW_WIDTH = 340
_C.DATA.NEW_HEIGHT = 256
_C.DATA.NEW_LENGTH = 32
_C.DATA.NEW_STEP = 1
_C.DATA.NUM_CLASSES = 400
_C.DATA.TEN_CROP = False
_C.DATA.THREE_CROP = False
_C.DATA.NUM_CROP = 1
_C.DATA.LOAD_VIDEOS = True
_C.DATA.AUGMENTATION = 'v1'
_C.DATA.NUM_SEGMENTS = 1
_C.DATA.TEMPORAL_JITTER = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
