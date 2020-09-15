
from yacs.config import CfgNode as CN

_C = CN()
_C.GPU = [0]
_C.LOG_INTERVAL = 500
_C.SAVE_DIR = 'runs'

# MODEL related params
_C.MODEL = CN()
# _C.MODEL.NAME = 'i3d_resnet50_v1_kinetics400'
# _C.MODEL.PRETRAINED = True
# _C.MODEL.RESUME = ''
# _C.MODEL.MODE = ''
_C.MODEL.EMB_SIZE = 512  # was num_units # Dimension of the embedding vectors and states
_C.MODEL.HIDDEN_SIZE = 2048  # Dimension of the hidden state in position-wise feed-forward networks
_C.MODEL.DROPOUT = 0.1
_C.MODEL.NUM_LAYERS = 6  # number of layers in the encoder and decoder
_C.MODEL.NUM_HEADS = 8  # number of heads in multi-head attention
_C.MODEL.SCALED_ATT = False  # use scale in attention
_C.MODEL.BLEU = 'tweaked'  # Schemes for computing bleu score. It can be: "tweaked": it uses similar steps in get_ende_bleu.sh in tensor2tensor repository, where compound words are put in ATAT format; "13a": This uses official WMT tokenization and produces the same results as official script (mteval-v13a.pl) used by WMT; "intl": This use international tokenization in mteval-v14a.pl

# DATASET related params
_C.DATA = CN()
_C.DATA.DATASET = 'MSVD'
_C.DATA.ROOT = 'datasets/MSVD'
_C.DATA.EXT = 'avi'
_C.DATA.SPLIT = 'train'
_C.DATA.SAMPLES_TYPE = 'captions'
_C.DATA.FEATURES = []
_C.DATA.BUCKET_SCHEME = 'constant'  # Strategy for generating bucket keys. It supports:  "constant": all the buckets have the same width;  "linear": the width of bucket increases linearly;  "exp": the width of bucket increases exponentially
_C.DATA.NUM_BUCKETS = 10
_C.DATA.BUCKET_RATIO = 0.0  # Ratio for increasing the throughput of the bucketing
_C.DATA.SRC_MAX_LEN = -1
_C.DATA.TGT_MAX_LEN = -1
_C.DATA.INPUT_SIZE = 224  # the model input size, transformed to this in the transform()
_C.DATA.CLIP_WIDTH = 340  # the clip width as read from video reader
_C.DATA.CLIP_HEIGHT = 256  # the clip height as read from video reader
_C.DATA.CLIP_LENGTH = 32  # the number of frames in a window (will be divided by CLIP_STRIDE)
_C.DATA.CLIP_STEP = -1  # the step between keyframes (-1 means every realtime second dependent on FPS)
_C.DATA.CLIP_STRIDE = 1  # the stride of frames in window
# _C.DATA.NUM_CLASSES = 400
# _C.DATA.TEN_CROP = False
# _C.DATA.THREE_CROP = False
# _C.DATA.NUM_CROP = 1
# _C.DATA.AUGMENTATION = 'v1'

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 10
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.BEAM_SIZE = 4
_C.TRAIN.LP_ALPHA = 0.6  # Alpha used in calculating the length penalty
_C.TRAIN.LP_K = 5  # K used in calculating the length penalty
_C.TRAIN.EPSILON = 0.1  # parameter for label smoothing
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 1.0
_C.TRAIN.WARMUP_STEPS = 4000
_C.TRAIN.NUM_ACCUMULATED = 1  # Number of steps to accumulate the gradients. This is useful to mimic large batch training with limited gpu memory
_C.TRAIN.MAGNITUDE = 3.0  # Magnitude of Xavier initialization
_C.TRAIN.AVG_CHECKPOINT = False  # Turn on to perform final testing based on the average of last few checkpoints
_C.TRAIN.NUM_AVERAGES = 5  # Perform final testing based on the average of last num_averages checkpoints
_C.TRAIN.AVG_START = 5  # Perform average SGD on last average_start epochs
_C.TRAIN.VAL_INTERVAL = 500


_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
