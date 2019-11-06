#!/bin/bash
cd datasets/Flickr30k

mkdir Entities
cd Entities

# Download and unzip
wget -c https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip
unzip -q annotations.zip

rm -rd Sentences

wget -c https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/train.txt
wget -c https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/val.txt
wget -c https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/test.txt

cd ../../..