#!/bin/bash

cd datasets
mkdir MSRVTT
cd MSRVTT

wget -O train_2017.zip -c "http://ms-multimedia-challenge.com/static/resource/train_2017.zip"

unzip -q train_2017.zip

rm train_2017.zip

mkdir videos

cd ../..