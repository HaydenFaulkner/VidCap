"""MSVD Dataset"""

import mxnet as mx
import os
from tqdm import tqdm

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats, concept_overlaps
from utils.video import extract_frames
from utils.text import id_to_syn, syn_to_word, parse, extract_nouns_verbs

__all__ = ['MSVD']


class MSVD(VisionDataset):
    """MSVD dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSVD'),
                 splits=['train'], transform=None, inference=False, every=25, subset=None):
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
        self._subset = subset

        self._videos_dir = os.path.join(self.root, 'videos')
        self._frames_dir = os.path.join(self.root, 'frames')

        # load the samples and labels at once
        self.samples, self.captions, self.mappings, self.frames = self._load_samples()

        if subset is not None:
            self.filter_on_objects(subset)

        self.img_ids = self.mappings.keys()
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
        video_id = self.sample_ids[ind]
        frame_paths = self.frames[video_id]
        cap = self.samples[video_id]['cap']

        frames = list()
        for frame_path in frame_paths:
            frames.append(mx.image.imread(frame_path))
        imgs = mx.nd.stack(*frames)

        if self._transform is not None:
            return self._transform(imgs, cap)

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
        frames = dict()

        # get the vid_id to youtube_id mappings
        with open(os.path.join(self.root, 'youtube_video_to_id_mapping.txt'), 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip().split() for line in lines]

        for yt_id, vid_id in lines:
            mappings[vid_id] = yt_id

            frames_dir = os.path.join(self._frames_dir, yt_id)
            if not os.path.exists(frames_dir):
                self.generate_frames(mappings={vid_id: yt_id})

            frames[vid_id] = sorted([f for f in os.listdir(frames_dir) if f[-4:] == '.jpg'])

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

        return samples, captions, mappings_split, frames

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

    def generate_frames(self, mappings=None):
        if mappings is None:
            mappings = self.mappings
        for k, v in tqdm(mappings.items(), desc='Generating Frames'):
            frames_dir = os.path.join(self._frames_dir, v)
            os.makedirs(frames_dir, exist_ok=True)
            video_path = os.path.join(self._videos_dir, v + '.avi')
            extract_frames(video_path, frames_dir)

    def filter_on_objects(self, names_file):
        assert names_file == 'filtered_det.tree', 'only filtered_det.tree accepted at this stage'
        with open(os.path.join('datasets', 'names', names_file), 'r') as f:
            lines = f.readlines()
        object_classes = set([l.split()[0] for l in lines])
        object_classes = set([syn_to_word(id_to_syn(wn_id)).replace('_', ' ') for wn_id in object_classes])

        objects = dict()
        for s in self.samples.values():
            vid = s['vid']
            # cap = s['cap'].split()
            cap = parse(s['cap'])
            _, nouns, _ = extract_nouns_verbs(cap)  # using nouns compared to just words gives subset aligned with stats

            exist = set.intersection(object_classes, set(nouns))
            if vid not in objects:
                objects[vid] = exist
            else:
                objects[vid] = objects[vid].union(exist)

        # need to remove these from the set
        remove_vids = set()
        for vid, objs in objects.items():
            if len(objs) < 1:
                remove_vids.add(vid)

        remove_samples = set()
        for k, v in self.samples.items():
            if v['vid'] in remove_vids:
                remove_samples.add(k)

        for vid in remove_vids:
            del self.captions[vid]
            del self.mappings[vid]
        for s in remove_samples:
            del self.samples[s]
        # below for debugs
        # for vid, objs in objects.items():
        #     print(vid, len(objs), objs)

        # counts = dict()
        # for vid, objs in objects.items():
        #     print(vid, len(objs), objs)
        #     if len(objs) not in counts:
        #         counts[len(objs)] = 1
        #     else:
        #         counts[len(objs)] += 1
        #
        # print('hist counts', counts)


if __name__ == '__main__':
    train_dataset = MSVD(splits=['train'], subset='filtered_det.tree')

    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'imagenetvid.synonyms'))
    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'filtered_det.tree'), use_synonyms=False, top=300)
    # print(overlaps)
    # print(missing)

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
