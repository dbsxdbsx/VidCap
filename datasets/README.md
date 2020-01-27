<h1 align='center'>Datasets</h1>

### PascalVOC Sentences
Consists of 1000 images from [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) with five 
captions per image. The dataset is presented [here](http://vision.cs.uiuc.edu/pascal-sentences/).

To **download** the dataset run `get_voc_dataset.sh` from the root dir:
```
VidCap$ python datasets/get_voc_dataset.sh
```

The script will download the [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) dataset and 
put the sentences into:
```
datasets/PascalVOC/VOCdevkit/Sentences
```

### Flickr 30k
Consists of 31783 images with five captions per image (158915 total). The dataset is available for [**download from 
Kaggle**](https://www.kaggle.com/hsankesara/flickr-image-dataset/downloads/flickr-image-dataset.zip/1) (requires sign in). 
Or the official website is [here](http://hockenmaier.cs.illinois.edu/DenotationGraph/), but you will need to fill out a 
form due to the copyright of the Flickr images, and the data **_might_** get sent to you.

To organise the dataset make a `Flickr30k` directory in `datasets`:
```
cd datasets
mkdir Flickr30k
```
and place the downloaded `flickr-image-dataset.zip` in it, resulting in:
```commandline
datasets/Flickr30k/flickr-image-dataset.zip
```

Then run `organise_flickr30k.sh` from the root dir:
```
VidCap$ . datasets/organise_flickr30k.sh
```

If you also want [**Flickr 30k Entities**](https://github.com/BryanPlummer/flickr30k_entities), which adds 244k 
coreference chains and 276k manually annotated bounding boxes, then follow up by running `get_flickr30k_entities.sh`:
```
VidCap$ . datasets/get_flickr30k_entities.sh
```

### MS Coco
To **download** the dataset run `get_coco_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_coco_dataset.sh
```

### ActivityNet
To **download** the dataset run `get_activitynet_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_activitynet_dataset.sh
```
This script will also attempt to download the videos from YouTube, note this can take a very long time and also that not
 all videos are still on YouTube. To get the full dataset instead you can fill out [**this form**](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform).
 
###  MSVD
Manually download the dataset from [dropbox](https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAC_TAsR7SkEYhkSdAFKcBlMa?dl=0)
 and save in a `MSVD` directory as `naacl15_translating_videos_processed_data.zip`:
```
datasets/MSVD/naacl15_translating_videos_processed_data.zip
```
Then run the `get_msvd_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_msvd_dataset.sh
```

### MSR-VTT
 
To **download** the training `.json` run `get_msrvtt_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_msrvtt_dataset.sh
```

To get the videos you can use the **mediafire.com** (*ew I know*) links below at your own risk:
- [train_val_annotation.zip](http://download1515.mediafire.com/t1cfuz3q7tdg/s88kuv5kqywpyym/train_val_annotation.zip)
- [test_videodatainfo.json.zip](http://download847.mediafire.com/egekeag8fowg/wvw68y9wmo3iw80/test_videodatainfo.json.zip)
- [train_val_videos.zip](http://download1079.mediafire.com/2xemo9i5s5jg/x3rrbe4hwp04e6w/train_val_videos.zip)
- [test_videos.zip](http://download876.mediafire.com/yf43j27femyg/czh8sezbo9s4692/test_videos.zip)
