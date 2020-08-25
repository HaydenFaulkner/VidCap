# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Customized dataloader for general video loading tasks.
Code adapted from https://raw.githubusercontent.com/dmlc/gluon-cv/master/gluoncv/data/video_custom/classification.py"""
import os
import numpy as np
from mxnet import nd
from mxnet.gluon.data import dataset

__all__ = ['VideoDataset']


class VideoDataset(dataset.Dataset):

    def __init__(self, cfg, transform=None):

        super(VideoDataset, self).__init__()

        from gluoncv.utils.filesystem import try_import_cv2, try_import_decord
        self.cv2 = try_import_cv2()
        self.root = cfg.DATA.ROOT
        self.split = cfg.DATA.SPLIT
        self.samples_type = cfg.DATA.SAMPLES_TYPE
        self.num_crop = cfg.DATA.NUM_CROP
        self.clip_width = cfg.DATA.CLIP_WIDTH
        self.clip_height = cfg.DATA.CLIP_HEIGHT
        self.clip_length = cfg.DATA.CLIP_LENGTH
        self.clip_step = cfg.DATA.CLIP_STEP
        self.clip_stride = cfg.DATA.CLIP_STRIDE
        self.target_height = cfg.DATA.INPUT_SIZE
        self.target_width = cfg.DATA.INPUT_SIZE
        self.video_ext = cfg.DATA.EXT
        self.slowfast = cfg.MODEL.NAME[:8] == 'slowfast'
        self.data_aug = cfg.DATA.AUGMENTATION
        self.transform = transform

        self.videos_dir = os.path.join(self.root, 'videos')

        assert self.split in ['train', 'val', 'test']
        assert self.samples_type in ['clips', 'frames']

        self.decord = try_import_decord()

        self.clips = self._populate_clips()
        if len(self.clips) == 0:
            raise(RuntimeError("Found 0 video clips, check your data directory."))

        self.video_meta = self._populate_meta()

        self.frames = self._populate_frames()

    def __getitem__(self, index):

        if self.samples_type == 'clips':
            vid_id, target = self.clips[index]
        else:
            vid_id, frame = self.frames[index]
            target = None  # todo target for frames

        vid_path = os.path.join(self.videos_dir, vid_id)

        if '.' in vid_path.split('/')[-1]:
            # data in the "setting" file already have extension, e.g., demo.mp4
            video_name = vid_path
        else:
            # data in the "setting" file do not have extension, e.g., demo
            # So we need to provide extension (i.e., .mp4) to complete the file name.
            video_name = '{}.{}'.format(vid_path, self.video_ext)

        # augment the data?
        if self.data_aug == 'v1':
            decord_vr = self.decord.VideoReader(video_name, width=self.clip_width, height=self.clip_height, num_threads=1)
        else:
            decord_vr = self.decord.VideoReader(video_name, num_threads=1)

        # get the frames for this window
        start_frame = frame - int(self.clip_length/2.0)
        end_frame = start_frame + self.clip_length
        # clip the window frame indexs (temporal padding with edge duplication)
        window_frames = [max(min(v, self.video_meta[vid_id]['duration']-1), 1) for v in list(range(start_frame, end_frame, self.clip_stride))]

        # if its the slowfast net we need to add some extras for the fast branch
        if self.slowfast:
            window_frames += [max(min(v, self.video_meta[vid_id]['duration']-1), 1) for v in list(range(start_frame+(self.clip_stride*4), end_frame, self.clip_stride*8))]  # clip the edges

        # extract the window frames from the clip
        try:
            video_data = decord_vr.get_batch(window_frames).asnumpy()
            clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(window_frames)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(window_frames, vid_id, self.video_meta[vid_id]['duration']))

        # perform any specified transforms
        if self.transform is not None:
            clip_input = self.transform(clip_input)

        # some input transforms
        if self.slowfast:
            sparse_samples = len(clip_input) // self.num_crop
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (sparse_samples, 3, self.target_height, self.target_width))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        else:
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (self.clip_length, 3, self.target_height, self.target_width))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

        # squeeze for 2D input case
        if self.clip_length == 1:
            clip_input = np.squeeze(clip_input, axis=2)

        return nd.array(clip_input), target, vid_id

    def __len__(self):
        if self.samples_type == 'clips':
            return len(self.clips)
        else:
            return len(self.frames)

    def _populate_clips(self):
        if not os.path.exists(self.root):
            raise(RuntimeError("Dataset root %s doesn't exist." % self.root))
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

    def _populate_meta(self):
        video_meta = dict()
        if os.path.exists(os.path.join(self.root, 'video_meta_'+self.split+'.txt')):
            with open(os.path.join(self.root, 'video_meta_'+self.split+'.txt'), 'r') as f:
                lines = f.readlines()
            lines = [line.rstrip().split() for line in lines]
            for vid_id, dur, fps in lines:
                video_meta[vid_id] = {'duration': int(dur), 'fps': int(fps)}
        else:
            for vid_id in set(self.videos):
                if vid_id not in video_meta:
                    video_meta[vid_id] = dict()
                vid_path = os.path.join(self.videos_dir, vid_id)

                if '.' in vid_path.split('/')[-1]:
                    video_name = vid_path
                else:
                    video_name = '{}.{}'.format(vid_path, self.video_ext)

                decord_vr = self.decord.VideoReader(video_name, num_threads=1)
                video_meta[vid_id]['path'] = video_name
                video_meta[vid_id]['duration'] = len(decord_vr)
                video_meta[vid_id]['fps'] = int(round(decord_vr.get_avg_fps()))

            with open(os.path.join(self.root, 'video_meta_'+self.split+'.txt'), 'w') as f:
                for vid_id in set(self.videos):
                    f.write("%s %d %d\n" % (vid_id, video_meta[vid_id]['duration'], video_meta[vid_id]['fps']))

        return video_meta

    def _populate_frames(self):
        frames = list()
        for vid_id in set(self.videos):
            if self.clip_step == -1:  # one every second
                vid_frames = list(range(self.video_meta[vid_id]['fps'], self.video_meta[vid_id]['duration'], self.video_meta[vid_id]['fps']))
            else:
                vid_frames = list(range(self.clip_step, self.video_meta[vid_id]['duration'], self.clip_step))
            for frame in vid_frames:
                frames.append((vid_id, frame))

        return frames
