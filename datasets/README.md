<h1 align='center'>Datasets</h1>

### PascalVOC Sentences
Consists of 1000 images from [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) with five 
captions per image. The dataset is presented [here](http://vision.cs.uiuc.edu/pascal-sentences/).

To **download** the dataset run `get_voc_sent_dataset.py` from the root dir:
```
python datasets/get_voc_sent_dataset.py
```

The script will download the images and sentences into:
```
datasets/PascalVOCSent/images
datasets/PascalVOCSent/sentences
```
