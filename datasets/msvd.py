"""MSVD Dataset"""

import os

from datasets.statistics import get_stats, concept_overlaps
from datasets.video import VideoDataset
from utils.text import id_to_syn, syn_to_word, parse, extract_nouns_verbs

__all__ = ['MSVD']


class MSVD(VideoDataset):
    """MSVD dataset."""

    def __init__(self, cfg, transform=None, inference=False, subset=None):
        """
        args:
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(MSVD, self).__init__(cfg, transform)

        self.captions = self._populate_captions()

        self.inference = inference
        self.subset = subset

        if subset is not None:
            self.filter_on_objects(subset)

        self.samples = self._determine_samples()

        if len(self.feature_list) > 0:
            print('Checking feature files exist...')
            self._check_features_exist()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    def __len__(self):
        return len(self.samples)

    def _populate_clips(self):
        clips = list()

        with open(os.path.join(self.root, 'sents_'+self.split+'_lc_nopunc.txt'), 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip().split('\t') for line in lines]

        with open(os.path.join(self.root, 'youtube_video_to_id_mapping.txt'), 'r') as f:
            mapping = f.readlines()
        mapping_tmp = [line.rstrip().split() for line in mapping]
        mapping = dict()
        for m in mapping_tmp:
            mapping[m[1]] = m[0]

        for vid_id, _ in lines:
            # each caption is a separate samples
            if mapping[vid_id] not in clips:
                clips.append(mapping[vid_id])

        return clips

    def _populate_captions(self):
        captions = dict()

        with open(os.path.join(self.root, 'sents_'+self.split+'_lc_nopunc.txt'), 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip().split('\t') for line in lines]

        with open(os.path.join(self.root, 'youtube_video_to_id_mapping.txt'), 'r') as f:
            mapping = f.readlines()
        mapping_tmp = [line.rstrip().split() for line in mapping]
        mapping = dict()
        for m in mapping_tmp:
            mapping[m[1]] = m[0]

        for vid_id, caption in lines:
            if mapping[vid_id] in captions.keys():
                captions[mapping[vid_id]].append(caption)
            else:
                captions[mapping[vid_id]] = [caption]

        return captions

    def video_captions(self, vid_id):
        return self.captions[vid_id]

    def image_captions(self, vid_id):
        return self.video_captions(vid_id)

    def video_ids(self):
        return self.clips

    def image_ids(self):
        return self.video_ids()

    def stats(self):
        return get_stats(self)

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
        for vid in self.clips:
            cap = self.captions[vid]
            _, nouns, _ = extract_nouns_verbs(parse(cap))  # using nouns compared to just words gives subset aligned with stats

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

        for vid in remove_vids:
            del self.captions[vid]

        self.clips = [v for v in self.clips if v not in remove_vids]
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

    from features.config import config as cfg

    cfg.DATA.DATASET = 'MSVD'
    cfg.DATA.ROOT = os.path.join('datasets', 'MSVD')
    cfg.DATA.EXT = 'avi'
    cfg.DATA.FEATURES = ['slowfast_4x16_resnet50_kinetics400', 'i3d_resnet50_v1_kinetics400']
    cfg.DATA.SAMPLES_TYPE = 'captions'

    for split in ['train', 'val', 'test']:
        cfg.DATA.SPLIT = split
        dataset = MSVD(cfg)
        for d in dataset:
            print(d)
            break
        print('-'*10 + '  ' + split + '  ' + '-'*10)
        print(dataset.stats())

    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'imagenetvid.synonyms'))
    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'filtered_det.tree'), use_synonyms=False, top=300)
    # print(overlaps)
    # print(missing)
