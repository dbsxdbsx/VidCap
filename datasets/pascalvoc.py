"""PascalVOC Sentence Dataset"""

import mxnet as mx
import numpy as np
import os

try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et

from gluoncv.data.base import VisionDataset

from datasets.statistics import get_stats
from utils.text import parse

__all__ = ['Flickr30k']


class PascalVOCSentence(VisionDataset):
    """PascalVOC Sentence dataset."""

    def __init__(self, root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
                 splits=None, transform=None, inference=False):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/PascalVOC/VOCdevkit')
            splits (tuple): a list of splits as strings (default is ((2007, 'trainval'), (2012, 'trainval')))
            transform: the transform to apply to the image/video and label (default is None)
            inference (bool): are we doing inference? (default is False)
        """
        super(PascalVOCSentence, self).__init__(root)
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._inference = inference
        self._splits = splits

        # load the classes
        self.categories = self.load_categories()
        self.wn_categories = self.load_wn_categories()

        # load the samples and boxess at once
        self.sample_ids, self.boxes, self.captions, self.image_sizes = self._load_dataset()

        print()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    @staticmethod
    def load_categories():
        """
        Gets a list of category names as specified in the pascalvoc.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'pascalvoc.names')
        with open(names_file, 'r') as f:
            categories = [line.strip() for line in f.readlines()]

        return categories

    @staticmethod
    def load_wn_categories():
        """
        Gets a list of category names as specified in the pascalvoc_wn.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'pascalvoc_wn.names')
        with open(names_file, 'r') as f:
            wn_classes = [line.strip() for line in f.readlines()]
        return wn_classes

    def category_synonyms(self):
        """
        Get a dict of hand-chosen synonyms for each category, handpicked from the common caption words

        :return: dict keyed with categories containing list of synonyms
        """
        synonyms_file = os.path.join('datasets', 'names', 'pascalvoc.synonyms')
        with open(synonyms_file, 'r') as f:
            synonyms_ = [line.strip().split(',') for line in f.readlines()]
        synonyms = dict()
        for syns in synonyms_:
            assert syns[0] in self.categories
            synonyms[syns[0]] = syns[1:]
        return synonyms

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
        boxes = np.array(self.boxes[self.sample_ids[sind]])
        img = mx.image.imread(img_path, 1)

        if self._transform is not None:
            return self._transform(img, boxes)

        if self._inference:
            return img, boxes, sind
        else:
            return img, boxes

    def sample_path(self, sind):
        return os.path.join(self.root, 'Images', str(self.sample_ids[sind]) + '.jpg')

    def sample_captions(self, sid):
        return self.captions[sid]

    def sample_boxes(self, sid):
        return self.boxes[sid]

    def _load_dataset(self):
        """
        Load the dataset from file, we only load the 1000 samples that have captions

        :return: sample_ids, boxes, captions, img_sizes
        """

        boxes = dict()
        captions = dict()
        img_sizes = dict()

        txts = os.listdir(os.path.join(self.root, 'Sentences'))
        sample_ids = [sid.split('.')[0] for sid in txts]
        for sid in sample_ids:
            with open(os.path.join(self.root, 'Sentences', str(sid) + '.txt'), 'r') as f:
                lines = f.readlines()
            caps = [parse(l.rstrip()) for l in lines]
            captions[sid] = caps

            sample_boxes, img_size = self._load_boxes(sid)
            boxes[sid] = sample_boxes
            img_sizes[sid] = img_size

        return sample_ids, boxes, captions, img_sizes

    def _load_boxes(self, sid):
        """
        Load boxes for a sample from the xml files

        :param sid: the sample ID
        :return: the boxes (a list of lists), and the image size (w, h)
        """

        anno_path = os.path.join(self.root, 'VOC2012', 'Annotations', sid + '.xml')
        assert os.path.exists(anno_path), "%s doesn't exist" % anno_path

        root = et.parse(anno_path).getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        img_size = (width, height)
        boxes = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            assert cls_name in self.categories, "Class %s is not in the categories" % cls_name
            cls_id = self.categories.index(cls_name)
            xml_box = obj.find('bndbox')
            xmin = (int(xml_box.find('xmin').text) - 1)
            ymin = (int(xml_box.find('ymin').text) - 1)
            xmax = (int(xml_box.find('xmax').text) - 1)
            ymax = (int(xml_box.find('ymax').text) - 1)

            xmin, ymin, xmax, ymax = self._validate_box(xmin, ymin, xmax, ymax, width, height)

            boxes.append([xmin, ymin, xmax, ymax, cls_id])

        if len(boxes) < 1:  # if no boxes we add a -1
            boxes.append([-1, -1, -1, -1, -1, -1])

        return boxes, img_size

    @staticmethod
    def _validate_box(xmin, ymin, xmax, ymax, width, height):
        """
        Set box to be within image, useful for boundary cases
        """
        if not 0 <= xmin < width or not 0 <= ymin < height or not xmin < xmax <= width or not ymin < ymax <= height:
            xmin = min(max(0, xmin), width - 1)
            ymin = min(max(0, ymin), height - 1)
            xmax = min(max(xmin + 1, xmax), width)
            ymax = min(max(ymin + 1, ymax), height)

        return xmin, ymin, xmax, ymax

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

    dataset = PascalVOCSentence()

    print(dataset.stats())

    # for s in tqdm(dataset, desc='Test Pass of Dataset'):
    #     pass
