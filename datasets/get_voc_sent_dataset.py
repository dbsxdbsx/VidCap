"""
Modified from https://github.com/rupy/PascalSentenceDataset
Thanks Rupy, nice drum set
"""

import os
import requests

from pyquery import PyQuery
from tqdm import tqdm
from urllib.parse import urljoin


DATASET_URL = 'http://vision.cs.uiuc.edu/pascal-sentences/'


def download_images():

    dom = PyQuery(DATASET_URL)
    for img in tqdm(dom('img').items(), desc="Downloading images from %s" % DATASET_URL, total=1000):
        img_src = img.attr['src']
        category, img_file_name = os.path.split(img_src)

        # make category directories
        output_dir = 'Images'
        os.makedirs(output_dir, exist_ok=True)

        # download image
        output = os.path.join(output_dir, img_file_name)
        if img_src.startswith('http'):
            img_url = img_src
        else:
            img_url = urljoin(DATASET_URL, img_src)
        if os.path.isfile(output):
            print("Already downloaded, Skipping: %s" % output)
            continue
        with open(output, 'wb') as f:

            while True:
                result = requests.get(img_url)
                raw = result.content
                if result.status_code == 200:
                    f.write(raw)
                    break
                print("error occurred while fetching img")
                print("retry...")


def download_sentences():
    dom = PyQuery(DATASET_URL)
    # tbody disappears in pyquery DOM
    for tr in tqdm(dom('body>table>tr').items(), desc="Downloading sentences from %s" % DATASET_URL, total=1000):
        img_src = tr('img').attr['src']
        category, img_file_name = os.path.split(img_src)

        # make category directories
        output_dir = 'Sentences'
        os.makedirs(output_dir, exist_ok=True)

        # dowonload sentences
        head, tail = os.path.splitext(img_file_name)
        sentence_file_name = head + ".txt"
        output = os.path.join(output_dir, sentence_file_name)
        if os.path.isfile(output):
            print("Already downloaded, Skipping: %s" % output)
            continue
        with open(output, 'w') as f:
            for td in tr('table tr td').items():
                f.write(td.text() + "\n")


if __name__ == "__main__":

    # download_images()
    download_sentences()
