"""MSRVTT Dataset"""

import json
import os

from datasets.statistics import get_stats, concept_overlaps
from datasets.video import VideoDataset
from utils.text import id_to_syn, syn_to_word, parse, extract_nouns_verbs

__all__ = ['MSRVTT']


class MSRVTT(VideoDataset):
    """MSRVTT dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSRVTT'), inference=False, subset=None, *args, **kwargs):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSRVTT')
            inference (bool): are we doing inference? (default is False)
            subset (str): specify an object file to only have captions that include particular objects
        """
        super(MSRVTT, self).__init__(root, *args, **kwargs)

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

        # load the json
        with open(os.path.join(self.root, 'videodatainfo_2017.json'), 'r') as f:
            data = json.load(f)

        # extract the caption data
        for sent in data['sentences']:
            if sent['video_id'] in self.captions.keys():
                self.captions[sent['video_id']].append(sent['caption'])
            else:
                self.captions[sent['video_id']] = [sent['caption']]
                self.videos.append(sent['video_id'])

            clips.append((sent['video_id'], sent['caption']))

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
    train_dataset = MSRVTT(split='train')#, subset='filtered_det.tree', download_missing=True)

    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'imagenetvid.synonyms'))
    # overlaps, missing = concept_overlaps(train_dataset, os.path.join('datasets', 'names', 'filtered_det.tree'), use_synonyms=False, top=300)
    # print(overlaps)
    # print(missing)

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
