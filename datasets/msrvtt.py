"""MSRVTT Dataset"""

import json
import mxnet as mx
import os

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats
from utils.video import extract_frames

__all__ = ['MSRVTT']


class MSRVTT(VisionDataset):
    """MSRVTT dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSRVTT'),
                 splits=['train'], transform=None, inference=False, every=25):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(MSRVTT, self).__init__(root)
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

        imgs = extract_frames(vid_path, frames_dir=None, overwrite=False, start=-1, end=-1, every=self._every)
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
        return NotImplementedError

    def _load_samples(self):
        """Load the samples"""
        samples = dict()
        captions = dict()
        mappings = dict()

        # load the json
        with open(os.path.join(self.root, 'videodatainfo_2017.json'), 'r') as f:
            data = json.load(f)

        # extract the video data
        vid_id_to_samp_id = dict()
        for vid in data['videos']:
            if vid['split'] in self._splits:
                vid_id_to_samp_id[vid['video_id']] = len(samples.keys())
                samples[len(samples.keys())] = {'vid': vid['video_id'], 'url': vid['url'],
                                                'st': vid['start time'], 'et': vid['end time'],
                                                'cap': list()}
                captions[vid['video_id']] = list()
                mappings[vid['video_id']] = vid['url']

        # extract the caption data
        for sent in data['sentences']:
            if sent['video_id'] in vid_id_to_samp_id.keys():
                sid = vid_id_to_samp_id[sent['video_id']]
                samples[sid]['cap'].append(sent['caption'])
                captions[sent['video_id']].append(sent['caption'])

        return samples, captions, mappings

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


if __name__ == '__main__':
    train_dataset = MSRVTT(splits=['train'])

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
