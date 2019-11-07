"""Flickr 30k Dataset"""

import json
import mxnet as mx
import numpy as np
import os

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats

__all__ = ['Flickr30k']


class Flickr30k(VisionDataset):
    """Flickr30k dataset."""

    def __init__(self, root=os.path.join('datasets', 'Flickr30k'),
                 splits=('train',), transform=None, inference=False):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(Flickr30k, self).__init__(root)
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._inference = inference
        self._splits = splits

        # load the classes
        self.categories = self.load_categories()

        # load the samples and labels at once
        self.sample, self.boxes, self.captions, self.image_sizes = self._load_dataset()
        self.sample_ids = sorted(list(self.sample.keys()))

        print()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    @staticmethod
    def load_categories():
        """
        Gets a list of category names as specified in the flickr30k.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'flickr30k.names')
        with open(names_file, 'r') as f:
            categories = [line.strip() for line in f.readlines()]

        return categories

    # def load_wn_categories(self):
    #     """
    #     Gets a list of category names as specified in the flickr30k_wn.names file
    #
    #     Returns:
    #         list : a list of strings
    #
    #     """
    #     names_file = os.path.join('./datasets/names/coco_wn.names')
    #     with open(names_file, 'r') as f:
    #         wn_classes = [line.strip() for line in f.readlines()]
    #     return wn_classes

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, sind):
        """
        Get a sample from the dataset

        Args:
            sind (int): index of the sample in the dataset

        Returns:
            mxnet.NDArray: input image/volume
            numpy.ndarray: label
            int: idx (if inference=True)
        """
        img_path = self.sample_path(sind)
        label = np.array(self.boxes[self.sample_ids[sind]])
        img = mx.image.imread(img_path, 1)

        if self._transform is not None:
            return self._transform(img, label)

        if self._inference:
            return img, label, sind
        else:
            return img, label

    def sample_path(self, sind):
        return os.path.join(self.root, 'Images', str(self.sample_ids[sind]) + '.jpg')

    def sample_captions(self, sid):
        return self.captions[sid]

    def sample_boxes(self, sid):
        return self.boxes[sid]

    def _load_dataset(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        samples = dict()
        samples_rev = dict()
        boxes = dict()
        captions = dict()
        img_sizes = dict()
        with open(os.path.join(self.root, 'dataset_flickr30k.json'), 'r') as f:
            d = json.load(f)

        for sample in d['images']:
            if sample['split'] in self._splits:
                sid = sample['imgid']
                caps = list()
                for cap in sample['sentences']:
                    caps.append(' '.join(cap['tokens']))

                samples[sid] = int(sample['filename'].split('.')[0])
                samples_rev[samples[sid]] = sid
                captions[sid] = caps

        with open(os.path.join(self.root, 'flickr30k_cleaned_class.json'), 'r') as f:
            d = json.load(f)

        for sample in d['annotations']:
            sid = sample['image_id']
            if sid in samples_rev.keys():
                sboxes = []
                for cap in sample['captions']:
                    for i, box in enumerate(cap['process_bnd_box']):
                        sbox = [int(coord) for coord in box]
                        sbox.append(self.categories.index(cap['process_clss'][i]))
                        sboxes.append(sbox)
                boxes[samples_rev[sid]] = sboxes
                img_sizes[samples_rev[sid]] = [int(s) for s in sample['image_size']]

        for sid in samples.keys():
            if sid not in boxes.keys():
                boxes[sid] = [[-1, -1, -1, -1, -1]]

        return samples, boxes, captions, img_sizes

    def image_size(self, sid):
        return self.image_sizes[sid]

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
    train_dataset = Flickr30k(splits=['train'])

    print(train_dataset.stats())

    # for s in tqdm(train_dataset, desc='Test Pass of Training Set'):
    #     pass
