"""MS COCO object detection dataset - Edited from GluonCV coco dataset code"""

from gluoncv.data.mscoco.utils import try_import_pycocotools
from gluoncv.data.base import VisionDataset
from gluoncv.utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy
import mxnet as mx
import numpy as np
import os
from tqdm import tqdm
from datasets.statistics import get_stats

__all__ = ['COCO']


class COCO(VisionDataset):
    """MS COCO detection dataset."""

    def __init__(self, root=os.path.join('datasets', 'MSCoco'),
                 splits=('instances_train2017', 'captions_train2017'), transform=None, min_object_area=0,
                 allow_empty=True, use_crowd=True, inference=False):
        """
        Args:
            root (str): root file path of the dataset (default is 'datasets/MSCoco')
            splits (list): a list of splits as strings (default is ['instances_train2017'])
            transform: the transform to apply to the image/video and label (default is None)
            min_object_area (int): minimum accepted ground-truth area of box, if smaller ignored (default is 0)
            allow_empty (bool): include samples that don't have any labelled boxes? (default is True)
            use_crowd (bool): use boxes labeled as crowd instance? (default is True)
            inference (bool): are we doing inference? (default is False)
        """
        super(COCO, self).__init__(root)
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._min_object_area = min_object_area
        self._allow_empty = allow_empty
        self._use_crowd = use_crowd
        self._inference = inference
        self._splits = splits
                
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []

        self.categories = self.load_categories()
        self.wn_categories = self.load_wn_categories()

        # load the samples and labels at once
        self.sample_ids, self.samples, self.boxes, self.captions = self._load_jsons()

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n'

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files. \
                Please use single JSON dataset and evaluate one by one".format(len(self._coco)))
        return self._coco[0]

    @staticmethod
    def load_categories():
        """
        Gets a list of category names as specified in the coco.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'coco.names')
        with open(names_file, 'r') as f:
            categories = [line.strip() for line in f.readlines()]

        return categories

    @staticmethod
    def load_wn_categories():
        """
        Gets a list of category names as specified in the coco_wn.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('datasets', 'names', 'coco_wn.names')
        with open(names_file, 'r') as f:
            wn_categories = [line.strip() for line in f.readlines()]
        return wn_categories

    def category_synonyms(self):
        """
        Get a dict of hand-chosen synonyms for each category, handpicked from the common caption words

        :return: dict keyed with categories containing list of synonyms
        """
        synonyms_file = os.path.join('datasets', 'names', 'coco.synonyms')
        with open(synonyms_file, 'r') as f:
            synonyms_ = [line.strip().split(',') for line in f.readlines()]
        synonyms = dict()
        for syns in synonyms_:
            assert syns[0] in self.categories
            synonyms[syns[0]] = syns[1:]
        return synonyms

    @property
    def annotation_dir(self):
        """
        The subdir for annotations. Default is 'annotations'(coco default)
        For example, a coco format json file will be searched as
        'root/annotation_dir/xxx.json'
        You can override if custom dataset don't follow the same pattern
        """
        return 'annotations'

    def _parse_image_path(self, entry):
        """
        How to parse image dir and path from entry.

        Args:
            entry (dict): COCO entry, e.g. including width, height, image path, etc..

        Returns:
            str : absolute path for corresponding image.

        """
        dirname, filename = entry['coco_url'].split('/')[-2:]
        abs_path = os.path.join(self.root, dirname, filename)
        return abs_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): index of the sample in the dataset

        Returns:
            mxnet.NDArray: input image/volume
            numpy.ndarray: label
            int: idx (if inference=True)
        """
        img_path = self.sample_path(idx)
        label = np.array(self.boxes[self.sample_ids[idx]])
        img = mx.image.imread(img_path, 1)

        if self._transform is not None:
            return self._transform(img, label)

        if self._inference:
            return img, label, idx
        else:
            return img, label

    def sample_path(self, sind):
        return self.samples[sind]

    def sample_captions(self, sid):
        return self.captions[sid]

    def sample_boxes(self, sid):
        return self.boxes[sid]

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        image_ids = list()
        samples = dict()
        boxes = dict()
        captions = dict()

        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self.root, self.annotation_dir, split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)

            anno = os.path.join(self.root, self.annotation_dir, split.replace('instances', 'captions')) + '.json'
            _coco_cap = COCO(anno)
            self._coco.append(_coco_cap)

            categories = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not categories == self.categories:
                raise ValueError("Incompatible category names with COCO: ")
            assert categories == self.categories
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            for sid, entry in zip(sorted(_coco.getImgIds()), _coco.loadImgs(sorted(_coco.getImgIds()))):
                image_ids.append(sid)
                abs_path = self._parse_image_path(entry)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                box = self._check_load_bbox(_coco, entry)
                caption = self._load_captions(_coco_cap, entry)
                if not box and not caption:
                    print("%s doesn't have any boxes or captions" % sid)
                    continue
                samples[sid] = abs_path
                boxes[sid] = box
                captions[sid] = caption

        return image_ids, samples, boxes, captions

    @staticmethod
    def _load_captions(coco, entry):
        entry_id = entry['id']
        # fix pycocotools _isArrayLike which don't work for str in python3
        entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        caps = coco.loadAnns(ann_ids)
        caps = [cap['caption'] for cap in caps]
        return caps

    def _check_load_bbox(self, coco, entry):
        """Check and load ground-truth labels"""
        entry_id = entry['id']
        # fix pycocotools _isArrayLike which don't work for str in python3
        entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj['area'] < self._min_object_area:
                continue
            if obj.get('ignore', 0) == 1:
                continue
            if not self._use_crowd and obj.get('iscrowd', 0):
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] > 0 and xmax > xmin and ymax > ymin:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid])
        if not valid_objs:
            if self._allow_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append([-1, -1, -1, -1, -1])
        return valid_objs

    def image_size(self, sid):
        entry = self.coco.loadImgs(sid)[0]
        return entry['width'], entry['height']

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
    train_dataset = COCO(splits=['instances_train2017', 'instances_val2017'])

    print(train_dataset.stats())

