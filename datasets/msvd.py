"""MSVD Dataset"""

import os

from datasets.statistics import get_stats, concept_overlaps
from datasets.video import VideoDataset
from utils.text import id_to_syn, syn_to_word, parse, extract_nouns_verbs

__all__ = ['MSVD']


class MSVD(VideoDataset):
    """MSVD dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSVD'), inference=False, subset=None, *args, **kwargs):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSVD')
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(MSVD, self).__init__(root, video_ext='.avi', *args, **kwargs)

        self.inference = inference
        self.subset = subset

        if subset is not None:
            self.filter_on_objects(subset)

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self):
        clips = list()
        self.captions = dict()
        self.videos = list()

        with open(os.path.join(self.root, 'sents_'+self.split+'_lc_nopunc.txt'), 'r') as f:
            lines = f.readlines()
        lines = [line.rstrip().split('\t') for line in lines]

        for vid_id, caption in lines:
            if vid_id in self.captions.keys():
                self.captions[vid_id].append(caption)
            else:
                self.captions[vid_id] = [caption]
                self.videos.append(vid_id)

            # each caption is a separate samples
            clips.append((vid_id, caption))

        return clips

    def video_captions(self, vid_id):
        return self.captions[vid_id]

    def image_captions(self, vid_id):
        return self.video_captions(vid_id)

    def video_ids(self):
        return [v for v, c in self.clips]

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
        for vid, cap in self.clips:
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

        self.clips = [(v, c) for v, c in self.clips if v not in remove_vids]
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
    train_dataset = MSVD(split='train', subset='filtered_det.tree')

    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'imagenetvid.synonyms'))
    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'filtered_det.tree'), use_synonyms=False, top=300)
    # print(overlaps)
    # print(missing)

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
