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
                 splits=('train',), transform=None, inference=False, label='boxes'):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(Flickr30k, self).__init__(root)
        assert label in ['boxes', 'captions']
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._inference = inference
        self._splits = splits
        self._label = label

        # load the classes
        self.categories = self.load_categories()

        # load the samples and labels at once
        self.images, self.boxes, self.captions, self.image_sizes = self._load_dataset()

        self.samples = dict()
        for img_id in self.images.keys():
            if label == 'boxes':
                self.samples[len(self.samples.keys())] = {'img': img_id,
                                                          'boxs': self.boxes[img_id],
                                                          'caps': self.captions[img_id]}
            else:
                for cap in self.captions[img_id]:
                    self.samples[len(self.samples.keys())] = {'img': img_id,
                                                              'boxs': self.boxes[img_id],
                                                              'cap': cap}
        self.sample_ids = sorted(list(self.samples.keys()))

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

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            sind (int): index of the sample in the dataset

        Returns:
            mxnet.NDArray: input image/volume
            numpy.ndarray: label
            int: idx (if inference=True)
        """
        sample_id = self.sample_ids[idx]
        img_path = self.sample_path(sample_id)
        if self._label == 'boxes':
            label = np.array(self.boxes[sample_id])
        else:
            label = self.samples[sample_id]['cap']
        img = mx.image.imread(img_path, 1)

        if self._transform is not None:
            return self._transform(img, label)

        if self._inference:
            return img, label, idx
        else:
            return img, label

    def sample_path(self, smp_id):
        img_id = self.samples[smp_id]['img']
        return os.path.join(self.root, 'Images', str(self.images[img_id]) + '.jpg')

    def image_captions(self, img_id):
        return self.captions[img_id]

    def image_boxes(self, img_id):
        return self.boxes[img_id]

    def image_ids(self):
        return self.images.keys()

    def _load_dataset(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        images = dict()
        images_rev = dict()
        boxes = dict()
        captions = dict()
        img_sizes = dict()
        with open(os.path.join(self.root, 'dataset_flickr30k.json'), 'r') as f:
            d = json.load(f)

        for sample in d['images']:
            if sample['split'] in self._splits:
                img_id = sample['imgid']
                caps = list()
                for cap in sample['sentences']:
                    caps.append(' '.join(cap['tokens']))

                images[img_id] = int(sample['filename'].split('.')[0])
                images_rev[images[img_id]] = img_id
                captions[img_id] = caps

        with open(os.path.join(self.root, 'flickr30k_cleaned_class.json'), 'r') as f:
            d = json.load(f)

        for sample in d['annotations']:
            img_id = sample['image_id']
            if img_id in images_rev.keys():
                sboxes = []
                for cap in sample['captions']:
                    for i, box in enumerate(cap['process_bnd_box']):
                        sbox = [int(coord) for coord in box]
                        sbox.append(self.categories.index(cap['process_clss'][i]))
                        sboxes.append(sbox)
                boxes[images_rev[img_id]] = sboxes
                img_sizes[images_rev[img_id]] = [int(s) for s in sample['image_size']]

        for img_id in images.keys():
            if img_id not in boxes.keys():
                boxes[img_id] = [[-1, -1, -1, -1, -1]]

        return images, boxes, captions, img_sizes

    def image_size(self, smp_id):
        img_id = self.samples[smp_id]['img']
        return self.image_sizes[img_id]

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
