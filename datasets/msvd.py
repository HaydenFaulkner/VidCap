"""MSVD Dataset"""

import mxnet as mx
import os
from tqdm import tqdm

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats, concept_overlaps
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
        vid_path = self.video_path(sid)
        cap = self.samples[sid]['cap']

        # imgs = extract_frames(vid_path, frames_dir=None, overwrite=False, start=-1, end=-1, every=self._every)
        imgs = None
        # todo may need to be numpy or something here
        imgs = mx.nd.stack(*imgs)

        # if self._transform is not None:
        #     return self._transform(imgs, cap)

        if self._inference:
            return imgs, cap, ind
        else:
            return imgs, cap

    def sample_path(self, sid):
        vid_id = self.samples[sid]['vid']
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

        mappings_split = dict()
        for split in self._splits:
            with open(os.path.join(self.root, 'sents_'+split+'_lc_nopunc.txt'), 'r') as f:
                lines = f.readlines()
            lines = [line.rstrip().split('\t') for line in lines]

            for vid_id, caption in lines:
                if vid_id in captions.keys():
                    captions[vid_id].append(caption)
                else:
                    captions[vid_id] = [caption]

                samples[len(samples.keys())] = {'vid': vid_id, 'cap': caption}  # each individual caption is a sample
                mappings_split[vid_id] = mappings[vid_id]

        return samples, captions, mappings_split

    def image_captions(self, vid_id):
        return self.video_captions(vid_id)

    def video_ids(self):
        return self.mappings.keys()

    def image_ids(self):
        return self.video_ids()

    def stats(self):
        """
        Get the dataset statistics

        Returns:
            str: an output string with all of the information
            list: a list of counts of the number of boxes per class

        """
        out_str = get_stats(self)

        return out_str

    def generate_frames(self):
        for k, v in tqdm(self.mappings.items(), desc='Generating Frames'):
            frames_dir = os.path.join(self.root, 'frames', v)
            os.makedirs(frames_dir, exist_ok=True)
            video_path = os.path.join(self.root, 'videos', v + '.avi')
            extract_frames(video_path, frames_dir)


if __name__ == '__main__':
    train_dataset = MSVD(splits=['train'])
    train_dataset.generate_frames()

    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'imagenetvid.synonyms'))
    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'filtered_det.tree'), use_synonyms=False, top=300)
    # print(overlaps)
    # print(missing)

    # print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
