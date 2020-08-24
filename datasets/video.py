# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Customized dataloader for general video loading tasks.
Code adapted from https://raw.githubusercontent.com/dmlc/gluon-cv/master/gluoncv/data/video_custom/classification.py"""
import os
import numpy as np
from mxnet import nd
from mxnet.gluon.data import dataset

__all__ = ['VideoDataset']


class VideoDataset(dataset.Dataset):
    """Load your own video dataset.

    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    split : str, required.
        Either of ['train', 'val', 'test']
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordingly.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    new_width : int, default 340.
        Scale the width of loaded image to 'new_width' for later multiscale cropping and resizing.
    new_height : int, default 256.
        Scale the height of loaded image to 'new_height' for later multiscale cropping and resizing.
    target_width : int, default 224.
        Scale the width of transformed image to the same 'target_width' for batch forwarding.
    target_height : int, default 224.
        Scale the height of transformed image to the same 'target_height' for batch forwarding.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    transform : function, default None.
        A function that takes data and label and transforms them.
    slowfast : bool, default False.
        If set to True, use data loader designed for SlowFast network.
        Christoph Feichtenhofer, etal, SlowFast Networks for Video Recognition, ICCV 2019.
    slow_temporal_stride : int, default 16.
        The temporal stride for sparse sampling of video frames in slow branch of a SlowFast network.
    fast_temporal_stride : int, default 2.
        The temporal stride for sparse sampling of video frames in fast branch of a SlowFast network.
    data_aug : str, default 'v1'.
        Different types of data augmentation pipelines. Supports v1, v2, v3 and v4.
    """
    def __init__(self, cfg, transform=None):

        super(VideoDataset, self).__init__()

        from gluoncv.utils.filesystem import try_import_cv2, try_import_decord
        self.cv2 = try_import_cv2()
        self.root = cfg.DATA.ROOT
        self.split = cfg.DATA.SPLIT
        self.num_segments = cfg.DATA.NUM_SEGMENTS
        self.num_crop = cfg.DATA.NUM_CROP
        self.new_width = cfg.DATA.NEW_WIDTH
        self.new_height = cfg.DATA.NEW_HEIGHT
        self.new_length = cfg.DATA.NEW_LENGTH
        self.new_step = cfg.DATA.NEW_STEP
        self.skip_length = self.new_length * self.new_step
        self.target_height = cfg.DATA.INPUT_SIZE
        self.target_width = cfg.DATA.INPUT_SIZE
        self.temporal_jitter = cfg.DATA.TEMPORAL_JITTER
        self.video_ext = cfg.DATA.EXT
        self.slowfast = cfg.MODEL.NAME[:8] == 'slowfast'
        self.slow_temporal_stride = cfg.MODEL.SLOW_TEMP_STRIDE
        self.fast_temporal_stride = cfg.MODEL.FAST_TEMP_STRIDE
        self.data_aug = cfg.DATA.AUGMENTATION
        self.transform = transform

        self.videos_dir = os.path.join(self.root, 'videos')

        assert self.split in ['train', 'val', 'test']

        if self.slowfast:
            assert self.slow_temporal_stride % self.fast_temporal_stride == 0, \
                'slow_temporal_stride needs to be multiples of slow_temporal_stride, please set it accordinly.'
            assert not self.temporal_jitter, \
                'Slowfast dataloader does not support temporal jitter. Please set temporal_jitter=False.'
            assert self.new_step == 1, 'Slowfast dataloader only support consecutive frames reading, please set new_step=1.'

        self.decord = try_import_decord()

        self.clips = self._make_dataset()
        if len(self.clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + self.root + "\n"
                               "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):

        vid_id, target = self.clips[index]

        vid_path = os.path.join(self.videos_dir, vid_id)

        if '.' in vid_path.split('/')[-1]:
            # data in the "setting" file already have extension, e.g., demo.mp4
            video_name = vid_path
        else:
            # data in the "setting" file do not have extension, e.g., demo
            # So we need to provide extension (i.e., .mp4) to complete the file name.
            video_name = '{}.{}'.format(vid_path, self.video_ext)

        if self.data_aug == 'v1':
            decord_vr = self.decord.VideoReader(video_name, width=self.new_width, height=self.new_height, num_threads=1)
        else:
            decord_vr = self.decord.VideoReader(video_name, num_threads=1)
        duration = len(decord_vr)

        if self.split == 'train':
            segment_indices, skip_offsets = self._sample_train_indices(duration)
        elif self.split == 'val':
            segment_indices, skip_offsets = self._sample_val_indices(duration)
        else:
            segment_indices, skip_offsets = self._sample_test_indices(duration)

        # N frames of shape H x W x C, where N = num_oversample * num_segments * new_length
        if self.slowfast:
            clip_input = self._video_TSN_decord_slowfast_loader(vid_path, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = self._video_TSN_decord_batch_loader(vid_path, decord_vr, duration, segment_indices, skip_offsets)

        if self.transform is not None:
            clip_input = self.transform(clip_input)

        if self.slowfast:
            sparse_sampels = len(clip_input) // (self.num_segments * self.num_crop)
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, self.target_height, self.target_width))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        else:
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (self.new_length, 3, self.target_height, self.target_width))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

        if self.new_length == 1:
            clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

        return nd.array(clip_input), target, vid_id

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self):
        if not os.path.exists(self.root):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (self.root)))
        clips = []
        with open(self.root) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                # line format: video_path, video_label
                if len(line_info) < 3:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(self.root, line_info[0])
                target = int(line_info[2])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _sample_val_indices(self, num_frames):
        if num_frames > self.num_segments + self.skip_length - 1:
            tick = (num_frames - self.skip_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _sample_test_indices(self, num_frames):
        if num_frames > self.skip_length - 1:
            tick = (num_frames - self.skip_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_mmcv_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                try:
                    if offset + skip_offsets[i] <= duration:
                        vid_frame = video_reader[offset + skip_offsets[i] - 1]
                    else:
                        vid_frame = video_reader[offset - 1]
                except:
                    raise RuntimeError('Error occured in reading frames from video {} of duration {}.'.format(directory, duration))
                if self.new_width > 0 and self.new_height > 0:
                    h, w, _ = vid_frame.shape
                    if h != self.new_height or w != self.new_width:
                        vid_frame = self.cv2.resize(vid_frame, (self.new_width, self.new_height))
                vid_frame = vid_frame[:, :, ::-1]
                sampled_list.append(vid_frame)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return sampled_list

    def _video_TSN_decord_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                try:
                    if offset + skip_offsets[i] <= duration:
                        vid_frame = video_reader[offset + skip_offsets[i] - 1].asnumpy()
                    else:
                        vid_frame = video_reader[offset - 1].asnumpy()
                except:
                    raise RuntimeError('Error occured in reading frames from video {} of duration {}.'.format(directory, duration))
                sampled_list.append(vid_frame)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return sampled_list

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list

    def _video_TSN_decord_slowfast_loader(self, directory, video_reader, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            fast_id_list = []
            slow_id_list = []
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1

                if (i + 1) % self.fast_temporal_stride == 0:
                    fast_id_list.append(frame_id)

                    if (i + 1) % self.slow_temporal_stride == 0:
                        slow_id_list.append(frame_id)

                if offset + self.new_step < duration:
                    offset += self.new_step

            fast_id_list.extend(slow_id_list)
            frame_id_list.extend(fast_id_list)
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list,
                                                                                                         directory,
                                                                                                         duration))
        return sampled_list
