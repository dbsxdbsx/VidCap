#!/bin/bash
cd datasets/Flickr30k

# Unzip the zip
unzip -q flickr-image-dataset.zip

mkdir Images
mv flickr30k_images/flickr30k_images/* Images/
mv flickr30k_images/results.csv results.csv
rm -dr flickr30k_images

# get Kapathy's neuralbabytalk annotations and splits
wget -c http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip -q caption_datasets.zip
rm dataset_coco.json
rm dataset_flickr8k.json

wget -O flickr30k_cleaned_class.json.tar.gz "https://ucabfca5cf79233090781a8412ba.dl.dropboxusercontent.com/cd/0/get/Ar2YASWwvYCQ5Ryl60vpxF9c3Ygh8RWVHIFg1_NQg5aK03b9ZpizO4YdRoJCMVhEMNUCbG5Z-5XdsnQ3op_n5d26p5GDNPL-zqm3vc6XJeh4Vg/file?_download_id=093594233167012992825650111962294540766301378483563320847102607205&_notify_domain=www.dropbox.com&dl=1"
tar xzf flickr30k_cleaned_class.json.tar.gz

cd ../..