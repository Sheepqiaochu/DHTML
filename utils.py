import os
from math import ceil

import cv2
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
from torchvision.datasets.utils import download_and_extract_archive

CALTECH_PIC = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
CALTECH_LABEL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar"


def download(dir, url, args, dist=None):
    if args.dataset == 'lfw':
        dist = dist if dist else url.split('/')[-1]
        print('Start to Download {} to {} from {}'.format(dist, dir, url))
        download_path = os.path.join(dir, dist)
        if os.path.isfile(download_path):
            print('File {} already downloaded'.format(download_path))
            return download_path
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 * 1024

        with open(download_path, 'wb') as f:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=ceil(total_size // block_size),
                    unit='MB', unit_scale=True):
                f.write(data)
        print('Downloaded {}'.format(dist))
        return download_path
    else:
        dist = dist if dist else url.split('/')[-1]
        print('Start to Download {} to {} from {}'.format(dist, dir, CALTECH_PIC))
        download_path = os.path.join(dir, dist)
        if os.path.isfile(download_path):
            print('File {} already downloaded'.format(download_path))
            return download_path
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 * 1024

        with open(download_path, 'wb') as f:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=ceil(total_size // block_size),
                    unit='MB', unit_scale=True):
                f.write(data)
        print('Downloaded {}'.format(dist))
        return download_path


def image_loader(image_path):
    return cv2.imread(image_path)


def generate_roc_curve(fpr, tpr, path):
    assert len(fpr) == len(tpr)

    fig = plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    fig.savefig(path, dpi=fig.dpi)
