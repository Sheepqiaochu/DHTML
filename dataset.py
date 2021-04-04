import os
import tarfile
from math import ceil, floor

from torch.utils import data
from utils import image_loader, download

DATASET_TARBALL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
PAIRS_TRAIN = "http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt"
PAIRS_VAL = "http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt"

CALTECH_PIC = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
CALTECH_LABEL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar"


def create_datasets(dataroot, args, train_val_split=0.5):
    if not os.path.isdir(dataroot):  # path of dataset(~/datasets/lfw|caltech)
        os.mkdir(dataroot)

    dataroot_files = os.listdir(dataroot)  # names of people
    if args.dataset == 'lfw':
        data_tarball_file = DATASET_TARBALL.split('/')[-1]  # assign 'lfw-deepfunneled.tgz' to data_tarball_file
        data_dir_name = data_tarball_file.split('.')[0]  # assign 'lfw-deepfunneled' to data_dir_name
    else:
        data_tarball_file = CALTECH_PIC.split('/')[-1]  # assign 'lfw-deepfunneled.tgz' to data_tarball_file
        data_dir_name = data_tarball_file.split('.')[0]  # assign 'lfw-deepfunneled' to data_dir_name

    if data_dir_name not in dataroot_files:
        if data_tarball_file not in dataroot_files:
            tarball = download(dataroot, DATASET_TARBALL, args)
        with tarfile.open(tarball, 'r') as t:
            t.extractall(dataroot)
    if args.dataset == 'lfw':
        images_root = os.path.join(dataroot, 'lfw-deepfunneled')
    else:
        images_root = os.path.join(dataroot, '101_ObjectCategories')

    names = os.listdir(images_root)  # names of people
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    training_set = []
    validation_set = []
    # mark class with number: 1 2 3 4 etc.
    for klass, name in enumerate(names):
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return image_path, klass, name

        images_of_person = os.listdir(os.path.join(images_root, name))  # path of exact people(like adam)
        total = len(images_of_person)  # count of people

        training_set += map(
            add_class,
            images_of_person[:ceil(total * train_val_split)])  # 90% of whole pictures(path of exact people(like adam))
        validation_set += map(
            add_class,
            images_of_person[floor(total * train_val_split):])  # rest of the pictures(path)

    return training_set, validation_set, len(names)


class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])  # images of train set
        if self.transform:
            image = self.transform(image)
        #  training_set, validation_set, len(names)
        return image, self.datasets[index][1], self.datasets[index][2]


class PairedDataset(data.Dataset):

    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader

        self.image_names_a = []
        self.image_names_b = []
        self.matches = []

        self._prepare_dataset()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        return (self.transform(self.loader(self.image_names_a[index])),
                self.transform(self.loader(self.image_names_b[index])),
                self.matches[index])

    def _prepare_dataset(self):
        raise NotImplementedError


class LFWPairedDataset(PairedDataset):

    def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)

        for pair in pairs:
            if len(pair) == 3:  # same person
                match = True
                # index1, index2 is the index of pictures of one man
                name1, name2, index1, index2 = \
                    pair[0], pair[0], int(pair[1]), int(pair[2])

            else:  # from two persons
                match = False
                name1, name2, index1, index2 = \
                    pair[0], pair[2], int(pair[1]), int(pair[3])
            # path of the picture
            self.image_names_a.append(os.path.join(
                self.dataroot, 'lfw-deepfunneled',
                name1, "{}_{:04d}.jpg".format(name1, index1)))

            self.image_names_b.append(os.path.join(
                self.dataroot, 'lfw-deepfunneled',
                name2, "{}_{:04d}.jpg".format(name2, index2)))
            self.matches.append(match)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            # first line: count number, read from second line
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs
