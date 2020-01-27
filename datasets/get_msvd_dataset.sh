#!/bin/bash

cd datasets
mkdir MSVD
cd MSVD

# Manually download from dropbox first!! https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAC_TAsR7SkEYhkSdAFKcBlMa?dl=0
unzip -q naacl15_translating_videos_processed_data.zip

rm naacl15_translating_videos_processed_data.zip

rm -r Models

# Download videos tar
wget -O videos.tar -c "http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"

tar xf videos.tar

mv YouTubeClips videos

rm videos.tar

cd ../..