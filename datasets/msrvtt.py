"""MSRVTT Dataset"""

import json
import mxnet as mx
import os
from tqdm import tqdm

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats, concept_overlaps
from utils.text import id_to_syn, syn_to_word, parse, extract_nouns_verbs
from utils.video import extract_frames

__all__ = ['MSRVTT']


class MSRVTT(VisionDataset):
    """MSRVTT dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSRVTT'),
                 splits=['train'], transform=None, inference=False, every=25, subset=None):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSRVTT')
            splits (list): a list of splits as strings (default is ['train'])
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
            every (int): keep only every n frames
        """
        super(MSRVTT, self).__init__(root)
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._inference = inference
        self._splits = splits
        self._every = every
        self._subset = subset

        self._videos_dir = os.path.join(self.root, 'videos')
        self._frames_dir = os.path.join(self.root, 'frames')

        # load the samples and labels at once
        self.samples, self.captions, self.mappings, self.videos = self._load_samples()

        # todo add self.frames param, maybe in load samples function?

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
        return os.path.join(self.root, 'videos', self.mappings[vid_id])  # + '.avi')

    def _load_samples(self):
        """Load the samples"""
        samples = dict()
        captions = dict()
        mappings = dict()

        # load the json
        with open(os.path.join(self.root, 'videodatainfo_2017.json'), 'r') as f:
            data = json.load(f)

        # extract the video data
        videos = dict()
        for vid in data['videos']:
            if vid['split'] in self._splits:
                videos[vid['video_id']] = {'url': vid['url'], 'st': vid['start time'], 'et': vid['end time']}
                mappings[vid['video_id']] = vid['url']
                captions[vid['video_id']] = list()

        # extract the caption data
        for sent in data['sentences']:
            samples[len(samples.keys())] = {'vid': sent['video_id'], 'cap': sent['caption']}
            captions[sent['video_id']].append(sent['caption'])

        return samples, captions, mappings, videos

    def video_captions(self, vid_id):
        return self.captions[vid_id]

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
        """
        Generate individual frame files for this dataset

        :param mappings: needed if isn't yet initialised for the object instance
        :return: None, but saves frames in self._frames_dir
        """
        if mappings is None:
            mappings = self.mappings
        assert mappings is not None
        for k, v in tqdm(mappings.items(), desc='Generating Frames'):
            frames_dir = os.path.join(self._frames_dir, v)
            os.makedirs(frames_dir, exist_ok=True)
            video_path = os.path.join(self._videos_dir, v + '.avi')
            extract_frames(video_path, frames_dir)

    def filter_on_objects(self, names_file):
        """
        Filter out samples that don't include an object in the tree specified in the names_file

        :param names_file: the .tree file
        :return: None, changes the object instance inner values
        """
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
    train_dataset = MSRVTT(splits=['train'], subset='filtered_det.tree')

    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'imagenetvid.synonyms'))
    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'filtered_det.tree'), use_synonyms=False, top=300)
    # print(overlaps)
    # print(missing)

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
