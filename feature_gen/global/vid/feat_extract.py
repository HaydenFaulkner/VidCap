'''
Feature extraction code taken from GluonCV:
https://gluon-cv.mxnet.io/build/examples_action_recognition/feat_custom.html
'''

import os
import sys
import time
import argparse
import logging
import math
import gc
import json

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video as video_transforms
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord

from datasets.video import VideoDataset
from datasets.msvd import MSVD
from datasets.msrvtt import MSRVTT


def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from pre-trained models for video related tasks.')
    parser.add_argument('--dataset', type=str, default='MSVD',
                        help='the root path to your data')
    parser.add_argument('--need-root', action='store_true',
                        help='if set to True, --data-dir needs to be provided as the root path to find your videos.')
    parser.add_argument('--data-list', type=str, default='',
                        help='the list of your data. You can either provide complete path or relative path.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='number of gpus to use. Use -1 for CPU')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                        help='enable using pretrained model from GluonCV.')
    parser.add_argument('--hashtag', type=str, default='',
                        help='hashtag for pretrained models.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--new-height', type=int, default=256,
                        help='new height of the resize image. default is 256')
    parser.add_argument('--new-width', type=int, default=340,
                        help='new width of the resize image. default is 340')
    parser.add_argument('--new-length', type=int, default=32,
                        help='new length of video sequence. default is 32')
    parser.add_argument('--new-step', type=int, default=1,
                        help='new step to skip video sequence. default is 1')
    parser.add_argument('--num-classes', type=int, default=400,
                        help='number of classes.')
    parser.add_argument('--ten-crop', action='store_true',
                        help='whether to use ten crop evaluation.')
    parser.add_argument('--three-crop', action='store_true',
                        help='whether to use three crop evaluation.')
    parser.add_argument('--video-loader', action='store_true', default=True,
                        help='if set to True, read videos directly instead of reading frames.')
    parser.add_argument('--use-decord', action='store_true', default=True,
                        help='if set to True, use Decord video loader to load data.')
    parser.add_argument('--slowfast', action='store_true',
                        help='if set to True, use data loader designed for SlowFast network.')
    parser.add_argument('--slow-temporal-stride', type=int, default=16,
                        help='the temporal stride for sparse sampling of video frames for slow branch in SlowFast network.')
    parser.add_argument('--fast-temporal-stride', type=int, default=2,
                        help='the temporal stride for sparse sampling of video frames for fast branch in SlowFast network.')
    parser.add_argument('--num-crop', type=int, default=1,
                        help='number of crops for each image. default is 1')
    parser.add_argument('--data-aug', type=str, default='v1',
                        help='different types of data augmentation pipelines. Supports v1, v2, v3 and v4.')
    parser.add_argument('--num-segments', type=int, default=1,
                        help='number of segments to evenly split the video.')
    opt = parser.parse_args()
    return opt


def extract_video_features(logger):
    opt = parse_args()
    logger.info(opt)
    gc.set_threshold(100, 5, 5)

    # set env
    if opt.gpu_id == -1:
        context = mx.cpu()
    else:
        gpu_id = opt.gpu_id
        context = mx.gpu(gpu_id)

    # get data preprocess
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video_transforms.VideoTenCrop(opt.input_size),
            video_transforms.VideoToTensor(),
            video_transforms.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 10
    elif opt.three_crop:
        transform_test = transforms.Compose([
            video_transforms.VideoThreeCrop(opt.input_size),
            video_transforms.VideoToTensor(),
            video_transforms.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 3
    else:
        transform_test = video_transforms.VideoGroupValTransform(size=opt.input_size,
                                                                 mean=image_norm_mean, std=image_norm_std)
        opt.num_crop = 1

    # get model
    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    classes = opt.num_classes
    model_name = opt.model
    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    feat_ext=True, num_segments=opt.num_segments, num_crop=opt.num_crop)
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.resume_params is not '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (opt.resume_params))
    else:
        logger.info('Pre-trained model is successfully loaded from the model zoo.')
    logger.info("Successfully built model {}".format(model_name))

    # build a pseudo dataset instance to use its children class methods
    if opt.dataset == 'MSVD':
        dataset = MSVD(num_segments=opt.num_segments,
                       num_crop=opt.num_crop,
                       new_length=opt.new_length*opt.fast_temporal_stride, #trying to fix for FTS>1
                       new_step=opt.new_step,
                       new_width=opt.new_width,
                       new_height=opt.new_height,
                       use_decord=opt.use_decord,
                       slowfast=opt.slowfast,
                       slow_temporal_stride=opt.slow_temporal_stride,
                       fast_temporal_stride=opt.fast_temporal_stride,
                       transform=transform_test)#data_aug=opt.data_aug)
    elif opt.dataset == 'MSRVTT':
        dataset = MSRVTT(num_segments=opt.num_segments,
                         num_crop=opt.num_crop,
                         new_length=opt.new_length*opt.fast_temporal_stride, #trying to fix for FTS>1
                         new_step=opt.new_step,
                         new_width=opt.new_width,
                         new_height=opt.new_height,
                         use_decord=opt.use_decord,
                         slowfast=opt.slowfast,
                         slow_temporal_stride=opt.slow_temporal_stride,
                         fast_temporal_stride=opt.fast_temporal_stride,
                         transform=transform_test)#data_aug=opt.data_aug)
    else:
        return NotImplementedError
        # root_dir = opt.dataset
    save_dir = os.path.join('datasets', opt.dataset, 'features', 'video', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # get data
    logger.info('Load %d video samples.' % len(dataset))

    start_time = time.time()
    for i, (video, cap, video_id) in enumerate(dataset):
        video_input = video.as_in_context(context)
        video_feat = net(video_input.astype(opt.dtype, copy=False))

        np.save(os.path.join(save_dir,  video_id), video_feat.asnumpy())

        if i > 0 and i % opt.log_interval == 0:
            logger.info('%04d/%04d is done' % (i, len(dataset)))

    end_time = time.time()
    logger.info('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    extract_video_features(logger)
