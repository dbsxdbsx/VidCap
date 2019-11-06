<h1 align='center'>Datasets</h1>

### PascalVOC Sentences
Consists of 1000 images from [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) with five 
captions per image. The dataset is presented [here](http://vision.cs.uiuc.edu/pascal-sentences/).

To **download** the dataset run `get_voc_sent_dataset.py` from the root dir:
```
VidCap$ python datasets/get_voc_sent_dataset.py
```

The script will download the images and sentences into:
```
datasets/PascalVOCSent/Images
datasets/PascalVOCSent/Sentences
```

### Flickr 30k
Consists of 31783 images with five captions per image (158915 total). The dataset is available for [**download from 
Kaggle**](https://www.kaggle.com/hsankesara/flickr-image-dataset/downloads/flickr-image-dataset.zip/1) (requires sign in). 
Or the official website is [here](http://hockenmaier.cs.illinois.edu/DenotationGraph/), but you will need to fill out a 
form due to the copyright of the Flickr images, and the data **_might_** get sent to you.

To organise the dataset make a `Flickr30k` directory in `datasets`:
```commandline
cd datasets
mkdir Flickr30k
```
and place the downloaded `flickr-image-dataset.zip` in it, resulting in:
```commandline
datasets/Flickr30k/flickr-image-dataset.zip
```

Then run `organise_flickr30k.sh` from the root dir:
```commandline
VidCap$ . datasets/organise_flickr30k.sh
```

If you also want [**Flickr 30k Entities**](https://github.com/BryanPlummer/flickr30k_entities), which adds 244k 
coreference chains and 276k manually annotated bounding boxes, then follow up by running `get_flickr30k_entities.sh`:
```commandline
VidCap$ . datasets/get_flickr30k_entities.sh
```