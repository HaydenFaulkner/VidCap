"""MSVD Dataset"""

import mxnet as mx
import numpy as np
import os

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats
from utils.video import extract_frames

__all__ = ['MSVD']


class MSVD(VisionDataset):
    """MSVD dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSVD'),
                 splits=['train'], transform=None, inference=False, every=25):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(MSVD, self).__init__(root)
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._inference = inference
        self._splits = splits
        self._every = every

        # load the samples and labels at once
        self.samples, self.captions, self.mappings = self._load_samples()
        self.sample_ids = list(self.samples.keys())

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, ind):
        """
        Get a sample from the dataset

        Args:
            ind (int): index of the sample in the dataset

        Returns:
            mxnet.NDArray: input volume/video
            numpy.ndarray: caption
            int: idx (if inference=True)
        """
        sid = self.sample_ids[ind]
        vid_id = self.samples[sid]['vid']
        vid_path = self.video_path(vid_id)
        cap = self.samples[sid]['cap']

        imgs = extract_frames(vid_path, frames_dir=None, overwrite=False, start=-1, end=-1, every=self._every)
        # todo may need to be numpy or something here
        imgs = mx.nd.stack(*imgs)

        # if self._transform is not None:
        #     return self._transform(imgs, cap)

        if self._inference:
            return imgs, cap, ind
        else:
            return imgs, cap

    def video_path(self, vid_id):
        return os.path.join(self.root, 'videos', self.mappings[vid_id] + '.avi')

    def video_captions(self, vid_id):
        return self.captions[vid_id]

    def _load_samples(self):
        """Load the samples"""
        samples = dict()
        captions = dict()
        mappings = dict()

        # get the vid_id to youtube_id mappings
        with open(os.path.join(self.root, 'youtube_video_to_id_mapping.txt'), 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip().split() for line in lines]

        for yt_id, vid_id in lines:
            mappings[vid_id] = yt_id

        for split in self._splits:
            with open(os.path.join(self.root, 'sents_'+split+'_lc_nopunc.txt'), 'r') as f:
                lines = f.readlines()
            lines = [line.rstrip().split('\n') for line in lines]

            for vid_id, caption in lines:
                if vid_id in captions.keys():
                    captions[vid_id].append(caption)
                else:
                    captions[vid_id] = [caption]

                samples[len(samples.keys())] = {'vid': vid_id, 'cap': caption}  # each individual caption is a sample

        return samples, captions, mappings

    def stats(self):
        """
        Get the dataset statistics

        Returns:
            str: an output string with all of the information
            list: a list of counts of the number of boxes per class

        """
        out_str = get_stats(self)

        return out_str


if __name__ == '__main__':
    train_dataset = MSVD(splits=['train'])

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
