#!/bin/bash

cd datasets
mkdir ActivityNet
cd ActivityNet

# Download the v1.3 json
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json

# Download and unzip the captions
mkdir Captions
cd Captions
wget -c https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip -q captions.zip
rm captions.zip
cd ..

# Download the enitities .jsons
mkdir Entities
cd Entities
wget -c https://github.com/facebookresearch/ActivityNet-Entities/raw/master/data/anet_entities_cleaned_class_thresh50_test_skeleton.json
wget -c https://github.com/facebookresearch/ActivityNet-Entities/raw/master/data/anet_entities_cleaned_class_thresh50_trainval.json
wget -c https://github.com/facebookresearch/ActivityNet-Entities/raw/master/data/anet_entities_skeleton.txt
wget -c https://github.com/facebookresearch/ActivityNet-Entities/raw/master/data/anet_entities_trainval.json
wget -c https://github.com/facebookresearch/ActivityNet-Entities/raw/master/data/split_ids_anet_entities.json

cd ../../..

python datasets/download_activitynet.py  # if this ModuleNotFound errors make sure VidCap dir os on PYTHONPATH