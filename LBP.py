import argparse

import torch
from torch.utils.data import DataLoader

import main
from dataset import Dataset, create_datasets
from imageaug import transform_for_lbp_training

# settings for LBP
radius = 2
points = radius * 8


def lbp_extract(args):
    dataset_dir = main.get_dataset_dir(args)
    log_dir = main.get_log_dir(args)
    training_set, validation_set, num_classes = create_datasets(dataset_dir)
    training_dataset = Dataset(training_set, transform=transform_for_lbp_training((224, 224)))
    validation_dataset = Dataset(validation_set)

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )

    # for image_path, j, k in training_set:
    #     image = cv2.imread(image_path)
    #     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # plt.imshow(image_rgb)
    #     lbp = local_binary_pattern(image_gray, points, radius)
    #     edges = filters.sobel(lbp)
    #     plt.imshow(edges, plt.cm.gray)
    # for images, targets, names in training_dataloader:
    #     print("fuck")


parser = argparse.ArgumentParser(description='center loss example')
parser.add_argument('--data_root', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log_dir', type=str,
                    help='log directory')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--dataset_dir', type=str,
                    help='directory with lfw dataset'
                         ' (default: $HOME/datasets/lfw)')
parser.add_argument('--weights', type=str,
                    help='pretrained weights to load '
                         'default: ($LOG_DIR/resnet18.pth)')
parser.add_argument('--evaluate', type=str,
                    help='evaluate specified model on lfw dataset')
parser.add_argument('--pairs', type=str,
                    help='path of pairs.txt '
                         '(default: $DATASET_DIR/pairs.txt)')
parser.add_argument('--roc', type=str,
                    help='path of roc.png to generated '
                         '(default: $DATASET_DIR/roc.png)')
parser.add_argument('--verify-model', type=str,
                    help='verify 2 images of face belong to one person,'
                         'the param is the model to use')
parser.add_argument('--verify-images', type=str,
                    help='verify 2 images of face belong to one person,'
                         'split image pathes by comma')

if __name__ == '__main__':
    args = parser.parse_args()
    lbp_extract(args)
