"""ActivityNet Dataset"""

import json
import mxnet as mx
import numpy as np
import os
from tqdm import tqdm

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats
from utils.video import extract_frames

__all__ = ['ActivityNet']


class ActivityNet(VisionDataset):
    """ActivityNet dataset."""

    def __init__(self, root=os.path.join('datasets', 'ActivityNet'),
                 splits=['train'], transform=None, inference=False, every=25, label='events'):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(ActivityNet, self).__init__(root)
        assert label in ['events', 'captions']
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._inference = inference
        self._splits = splits
        self._every = every
        self._label = label

        # load the samples and labels at once
        self.videos, self.events, self.captions = self._load_samples()

        self.samples = dict()
        for vid_id in self.videos.keys():
            if label == 'events':  # todo could match the overlaps between captions and events so have one event and/or caption per sample
                for evt in self.events[vid_id]:
                    self.samples[len(self.samples.keys())] = {'vid': vid_id, 'split': self.videos[vid_id]['split'],
                                                              'evt': evt,
                                                              'caps': self.captions[vid_id]}
            else:
                for cap in self.captions[vid_id]:
                    if len(cap) > 0:  # when dataset is for captioning remove the few samples that dont have captions
                        self.samples[len(self.samples.keys())] = {'vid': vid_id, 'split': self.videos[vid_id]['split'],
                                                                  'evts': self.events[vid_id],
                                                                  'cap': cap}
        self.sample_ids = sorted(list(self.samples.keys()))


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
        vid_path = self.sample_path(sid)
        if self._label == 'captions':
            cap = self.samples[sid]['cap']
            start = cap['start']
            end = cap['end']
            lab = cap['cap']
        else:
            evt = self.samples[sid]['evt']
            start = evt['start']
            end = evt['end']
            lab = evt['label']

        # todo extract the correct frames for an event of a caption ...
        imgs = extract_frames(vid_path, frames_dir=None, overwrite=False, start=start, end=end, every=self._every,
                              seconds=True)
        imgs = mx.nd.array(np.concatenate([np.expand_dims(img, axis=0) for img in imgs], axis=0))

        # if self._transform is not None:
        #     return self._transform(imgs, cap)

        if self._inference:
            return imgs, lab, ind
        else:
            return imgs, lab

    def sample_path(self, sid):
        vid_id = self.samples[sid]['vid']
        split = self.samples[sid]['split']
        if split in ['training', 'validation']:
            split = 'train_val'
        else:
            split = 'test'
        return os.path.join(self.root, 'vids', 'v1-3', split, 'v_' + vid_id + '.mp4')

    def video_captions(self, vid_id):
        return [cap_data['cap'] for cap_data in self.captions[vid_id]]

    def _load_samples(self):
        """Load the samples"""
        videos = dict()
        events = dict()
        captions = dict()

        # load the events json
        with open(os.path.join(self.root, 'activity_net.v1-3.min.json'), 'r') as f:
            data = json.load(f)

        for split in self._splits:
            split = 'val_1' if split == 'val' else split
            if split != 'test':
                with open(os.path.join(self.root, 'Captions', split + '.json'), 'r') as f:
                    caption_data = json.load(f)

                for yt_id, info in caption_data.items():
                    captions[yt_id[2:]] = list()
                    for c, cap in enumerate(info['sentences']):
                        captions[yt_id[2:]].append({'start': info['timestamps'][c][0], 'end': info['timestamps'][c][1],
                                                    'cap': cap})

            split = 'training' if split == 'train' else split
            split = 'validation' if split in ['val_1', 'val_2', 'val'] else split
            split = 'testing' if split == 'test' else split

            no_caps = list()
            for yt_id, info in data['database'].items():
                if info['subset'] == split:
                    videos[yt_id] = {'duration': info['duration'], 'split': info['subset'],
                                     'w': info['resolution'].split('x')[0], 'h': info['resolution'].split('x')[1]}
                    for ann in info['annotations']:
                        if yt_id in events.keys():
                            events[yt_id].append({'start': ann['segment'][0], 'end': ann['segment'][1],
                                                  'label': ann['label']})
                        else:
                            events[yt_id] = [{'start': ann['segment'][0], 'end': ann['segment'][1],
                                              'label': ann['label']}]
                    if yt_id not in captions.keys():
                        captions[yt_id] = list()
                        no_caps.append(yt_id)

            print("%d videos [%s] have no captions" % (len(no_caps), ', '.join(no_caps)))

        return videos, events, captions

    def image_captions(self, vid_id):
        return self.video_captions(vid_id)

    def video_ids(self):
        return list(set([s['vid'] for s in self.samples.values()]))

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


if __name__ == '__main__':
    train_dataset = ActivityNet(splits=['val'], label='events')

    # print(train_dataset.stats())

    for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
        print(s)
        pass
