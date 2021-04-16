import os
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.ticker import MultipleLocator
from skimage.feature import local_binary_pattern


def plot_loss(epoch, path, loss):
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


def lbp_loader(image_path):
    radius = 2
    points = radius * 8
    img = cv2.imread(image_path)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image_gray, points, radius, method='var')

    return lbp


#
# lbp = lbp_loader('D:\\yangqiancheng\\Desktop\\pics\\Alexander_Downer_0001.jpg')
# cv2.imwrite('D:\\yangqiancheng\\Desktop\\pics\\Alexander_Downer_0001.jpg', lbp)

def acc_format(filename):
    acc = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()[0:]):
            if not (len(line) == '\t' or len(line) == '\n' or i % 2 == 1):
                acc.append(float('0' + line.strip().split('0')[1]))

    return acc


epoch = range(20, 1200, 40)
acc = acc_format('D:\\yangqiancheng\\Desktop\\pics\\source.txt')
acc2 = acc_format('D:\\yangqiancheng\\Desktop\\pics\\source_flexible.txt')
acc3 = acc_format('D:\\yangqiancheng\\Desktop\\pics\\source_fixed.txt')

plt.plot(epoch, acc[0:len(epoch)], color="r", linestyle="-", marker="^", linewidth=2, label='FGM')
plt.plot(epoch, acc2[0:len(epoch)], color="b", linestyle="-", marker="s", linewidth=2, label='origin flexible lr')
plt.plot(epoch, acc3[0:len(epoch)], color="g", linestyle="-", marker=".", linewidth=2, label='origin fixed lr')

x_major_locator = MultipleLocator(100)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.xlabel("Training epoch")
plt.ylabel("Testing accuracy")
plt.legend(by_label.values(), by_label.keys())
plt.savefig('D:\\yangqiancheng\\Desktop\\pics\\source_training', dpi=300)
plt.show()
