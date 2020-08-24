'''
Feature extraction code taken from GluonCV:
https://gluon-cv.mxnet.io/build/examples_action_recognition/feat_custom.html
'''

import gc
import mxnet as mx
import numpy as np
import os
import time

from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video as video_transforms
from gluoncv.model_zoo import get_model

from datasets.msvd import MSVD
from datasets.msrvtt import MSRVTT


def extract_video_features(cfg, log):

    gc.set_threshold(100, 5, 5)

    # set env
    if len(cfg.GPU) == 0:
        context = mx.cpu()
    else:
        context = mx.gpu(cfg.GPU[0])

    # get data preprocess
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if cfg.DATA.TEN_CROP:
        transform_test = transforms.Compose([
            video_transforms.VideoTenCrop(cfg.DATA.INPUT_SIZE),
            video_transforms.VideoToTensor(),
            video_transforms.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        cfg.DATA.NUM_CROP = 10
    elif cfg.DATA.THREE_CROP:
        transform_test = transforms.Compose([
            video_transforms.VideoThreeCrop(cfg.DATA.INPUT_SIZE),
            video_transforms.VideoToTensor(),
            video_transforms.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        cfg.DATA.NUM_CROP = 3
    else:
        transform_test = video_transforms.VideoGroupValTransform(size=cfg.DATA.INPUT_SIZE,
                                                                 mean=image_norm_mean, std=image_norm_std)
        cfg.DATA.NUM_CROP = 1

    # get model
    classes = cfg.DATA.NUM_CLASSES
    model_name = cfg.MODEL.NAME
    net = get_model(name=model_name, nclass=classes, pretrained=cfg.MODEL.PRETRAINED,
                    feat_ext=True, num_segments=cfg.DATA.NUM_SEGMENTS, num_crop=cfg.DATA.NUM_CROP)
    net.cast('float32')
    net.collect_params().reset_ctx(context)
    if cfg.MODEL.MODE == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if cfg.MODEL.RESUME is not '' and cfg.MODEL.PRETRAINED is '':
        net.load_parameters(cfg.MODEL.RESUME, ctx=context)
        log.info('Pre-trained model %s is successfully loaded.' % cfg.MODEL.RESUME)
    else:
        log.info('Pre-trained model is successfully loaded from the model zoo.')
    log.info("Successfully built model {}".format(model_name))

    # build a pseudo dataset instance to use its children class methods
    if cfg.DATA.DATASET == 'MSVD':
        dataset = MSVD(cfg, transform=transform_test)
    elif cfg.DATA.DATASET == 'MSRVTT':
        dataset = MSRVTT(cfg, transform=transform_test)
    else:
        return NotImplementedError

    save_dir = os.path.join('datasets', cfg.DATA.DATASET, 'features', 'video', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # get data
    log.info('Load %d video samples.' % len(dataset))

    start_time = time.time()
    for i, (video, cap, video_id) in enumerate(dataset):
        video_input = video.as_in_context(context)
        video_feat = net(video_input.astype('float32', copy=False))

        np.save(os.path.join(save_dir,  video_id), video_feat.asnumpy())

        if i > 0 and i % cfg.LOG_INTERVAL == 0:
            log.info('%04d/%04d is done' % (i, len(dataset)))

    end_time = time.time()
    log.info('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))
