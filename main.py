import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Dataset, create_datasets, LFWPairedDataset, UnlabeledDataset
from device import device
from imageaug import transform_for_infer, transform_for_training, transform_for_lbp
from metrics import compute_roc
from models import Resnet50FaceModel, Resnet18FaceModel
from models.ShuffleNet_Target import ShuffleNet_Target
from models.ShuffleNet_Source import ShuffleNet_Source
from models.MetricNet import MetricNet
from trainer import Trainer
from utils import download, generate_roc_curve, image_loader, lbp_loader
from utils import dataset_spilt


def main(args):
    if args.evaluate:
        evaluate(args)
    elif args.verify_model:
        verify(args)
    else:
        train(args)


def get_dataset_dir(args):
    home = os.path.expanduser("~")
    dataset_dir = args.dataset_dir if args.dataset_dir else os.path.join(
        home, 'datasets', 'lfw')

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    return dataset_dir


def get_log_dir(args):
    log_dir = args.log_dir if args.log_dir else os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logs')

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    return log_dir


def get_model_class(args):
    if args.arch == 'resnet18':
        model_class = Resnet18FaceModel
    if args.arch == 'resnet50':
        model_class = Resnet50FaceModel
    if args.arch == 'ShuffleNetV2':
        model_class = ShuffleNet_Target
    if args.arch == 'MetricNet':
        model_class = MetricNet
    return model_class


def load_model(args, name_counts):
    model1 = ShuffleNet_Source(width_mul=1.5, n_classes=name_counts).to(device)
    model2 = ShuffleNet_Target(width_mul=0.5, n_classes=name_counts).to(device)
    state_file1 = args.model1
    if not os.path.isfile(state_file1):
        raise RuntimeError(
            "resume file {} is not found".format(state_file1))

    check_point1 = torch.load(state_file1)
    print("loading checkpoint {}".format(state_file1))
    model1.load_state_dict(check_point1['state_dict'], strict=False)

    # state_file2 = args.model2
    # if not os.path.isfile(state_file2):
    #     raise RuntimeError(
    #         "resume file {} is not found".format(state_file2))

    # check_point2 = torch.load(state_file2)
    # print("loading checkpoint {}".format(state_file2))
    # model2.load_state_dict(check_point2['state_dict'], strict=False)

    return model1, model2


def lr_tune(epoch):
    if epoch < 150:
        return 1
    elif epoch < 400:
        return 0.1 * (pow(0.9, epoch // 40))
    else:
        return 0.03 * (pow(0.9, epoch // 64))


def train(args):
    # load log path
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)

    # split the dataset
    labeled_set, unlabeled_set, test_set = dataset_spilt()

    # load dataset
    labeled_dataset = Dataset(labeled_set, target_transform=transform_for_lbp((224, 224)))
    unlabeled_dataset = UnlabeledDataset(unlabeled_set, transform_for_training((224, 224)),
                                         target_transform=transform_for_lbp((224, 224)))
    test_set = Dataset(test_set, transform_for_infer((224, 224)))

    labeled_dataloader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    unlabeled_dataloader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    model1, model2 = load_model(args, 5749)
    trainables_wo_bn = [param for name, param in model2.named_parameters() if
                        param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model2.named_parameters() if
                          param.requires_grad and 'bn' in name]

    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.001},
        {'params': trainables_only_bn}
    ], lr=args.lr, momentum=0.9)

    learning_rate_epoch = lambda e: 1.0 * (pow(0.9, e / 64)) if e < 200 else (
        0.4 * (pow(0.9, e / 64)) if e < 400 else 0.1 * (pow(0.9, e / 64)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_tune,
        last_epoch=-1)

    trainer = Trainer(
        optimizer,
        scheduler,
        model1,
        model2,
        labeled_dataloader,
        unlabeled_dataloader,
        test_dataloader,
        sigma=args.sigma,
        phi=args.phi,
        max_epoch=args.epochs,
        resume=args.resume,
        log_dir=log_dir
    )
    trainer.train()


def evaluate(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    pairs_path = args.pairs if args.pairs else \
        os.path.join(dataset_dir, 'pairs.txt')

    if not os.path.isfile(pairs_path):
        download(dataset_dir, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    dataset = LFWPairedDataset(
        dataset_dir, pairs_path, transform_for_lbp(model_class.IMAGE_SHAPE), loader=lbp_loader)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    model = model_class(width_mul=args.width_mul).to(device)

    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    embedings_a = torch.zeros(len(dataset), model.FEATURE_DIMENSION)
    embedings_b = torch.zeros(len(dataset), model.FEATURE_DIMENSION)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) \
            in enumerate(dataloader):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        _, batched_embedings_a, _ = model(images_a)
        _, batched_embedings_b, _ = model(images_b)

        start = args.batch_size * iteration
        end = start + current_batch_size

        embedings_a[start:end, :] = batched_embedings_a.data
        embedings_b[start:end, :] = batched_embedings_b.data
        matches[start:end] = batched_matches.data

    thresholds = np.arange(0, 4, 0.1)
    distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)  # L2 distance
    print(distances)
    print(torch.mean(distances))
    tpr, fpr, accuracy, best_thresholds = compute_roc(
        distances,
        matches,
        thresholds
    )

    roc_file = args.roc if args.roc else os.path.join(log_dir, 'roc.png')
    generate_roc_curve(fpr, tpr, roc_file)
    print(args.evaluate)
    print('Model accuracy is {}'.format(accuracy))
    print('ROC curve generated at {}'.format(roc_file))


def verify(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    model = model_class(False).to(device)
    checkpoint = torch.load(args.verify_model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    image_a, image_b = args.verify_images.split(',')
    image_a = transform_for_infer(
        model_class.IMAGE_SHAPE)(image_loader(image_a))
    image_b = transform_for_infer(
        model_class.IMAGE_SHAPE)(image_loader(image_b))
    images = torch.stack([image_a, image_b]).to(device)

    _, (embedings_a, embedings_b) = model(images)

    distance = torch.sum(torch.pow(embedings_a - embedings_b, 2)).item()
    print("distance: {}".format(distance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='center loss example')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--log_dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--arch', type=str, default='MetricNet',
                        help='network arch to use, support resnet18 and '
                             'resnet50, shuffleNetV2 (default: shuffleNetV2)')
    parser.add_argument('--resume', type=str,
                        help='model path to the resume training',
                        default=False)
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
    parser.add_argument('--width_mul', type=float, default=1.0,
                        help='width_mul of the shuffleNet')
    parser.add_argument('--model1', type=str, default='logs/models/model1.pth.tar',
                        help='path of roc.png to generated '
                             '(default: $DATASET_DIR/roc.png)')
    parser.add_argument('--sigma', type=int, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--phi', type=int, metavar='N',
                        help='input batch size for training (default: 1000)')
    args = parser.parse_args()
    main(args)
