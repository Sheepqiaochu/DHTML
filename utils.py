import os
from collections import OrderedDict
from math import ceil
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.io as sio
from skimage.feature import local_binary_pattern
from tqdm import tqdm


def download(dir, url, dist=None):
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


def lbp_loader(image_path):
    radius = 2
    points = radius * 8
    img = cv2.imread(image_path)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image_gray, points, radius, method='uniform')
    lbp = lbp.astype(np.uint8)

    return lbp


def image_loader(image_path):
    return cv2.imread(image_path)


def generate_roc_curve(fpr, tpr, path):
    assert len(fpr) == len(tpr)

    fig = plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    fig.savefig(path, dpi=fig.dpi)


def dataset_spilt():
    count = 0
    path = {}
    unlabeled_set = []
    labeled_set = []
    test_set = []

    home = os.path.expanduser("~")
    data_root = os.path.join(home, 'datasets', 'lfw', 'lfw-deepfunneled')
    names = os.listdir(data_root)

    mat_data = sio.loadmat('LightenedCNN_C_lfw.mat')
    features_of_person = mat_data.get('features')
    label_of_person = mat_data.get('labels_original').astype(np.int)

    for name in names:
        image_root = os.path.join(data_root, name)
        no = os.listdir(image_root)
        for pic in no:
            path[count] = os.path.join(image_root, pic)
            count += 1

    for i in range(len(path)):
        k = np.random.randint(low=0, high=10)
        if k < 5:
            labeled_set.append([label_of_person[0][i].item(), path[i]])
            unlabeled_set.append([label_of_person[0][i].item(), features_of_person[i], path[i]])
        elif k > 5:
            unlabeled_set.append([label_of_person[0][i].item(), features_of_person[i], path[i]])
        else:
            test_set.append([label_of_person[0][i].item(), path[i]])

    return labeled_set, unlabeled_set, test_set


def load_features():
    training_set = []
    validation_set = []

    # load mat file
    mat_data = sio.loadmat('LightenedCNN_C_lfw.mat')
    label_of_person = mat_data.get('labels_original').astype(np.int)

    features_of_person = mat_data.get('features')
    total = len(features_of_person)
    for i in range(total):
        k = np.random.randint(low=0, high=10)
        if k <= 8:
            training_set.append([label_of_person[0][i].item(), features_of_person[i]])
        else:
            validation_set.append([label_of_person[0][i].item(), features_of_person[i]])
    # for i in range(ceil(total * train_val_split)):
    #     training_set.append([label_of_person[0][i].item(), features_of_person[i]])
    # for i in range(floor(total * train_val_split), total):
    #     validation_set.append([label_of_person[0][i].item(), features_of_person[i]])
    # print(training_set.shape)
    # print(validation_set.shape)
    # for images, targets in training_set:
    #     print(type(images))
    return training_set, validation_set


def plot_loss(epoch, path, loss):
    # loss = {
    #     "f1": [1, 2, 3, 4],
    #     "f2": [5, 6, 7, 8],
    #     "f3": [9, 10, 11, 12]
    # }
    plt_color = ['blue', 'red', 'yellow', 'green']

    data_path = os.path.join(path, 'loss_figure')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    file_path = os.path.join(data_path, str(epoch))
    figure_path = os.path.join(data_path, str(epoch) + '.jpg')

    torch.save(loss, file_path)
    loss = torch.load(file_path)
    for i, name in enumerate(loss):
        epochs = range(0, len(loss[name]))
        plt.plot(epochs, loss[name], color=plt_color[i], label=name)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(figure_path)
    plt.legend(by_label.values(), by_label.keys())
    # plt.show()

#
# for i in range(5):
#     plot_loss()
