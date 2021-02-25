import torch
from torch.nn import functional as F
from device import device
import scipy.io as sio


def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]  # center of the target
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)  # size(bs*features)
    targets, indices = torch.sort(targets)  # sort targets from small one to bigger one
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
        targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
        targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
        1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
        targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result


def get_distillation_loss(logits_source, logits_target):
    criterion = torch.nn.KLDivLoss(reduction='mean')
    T = 10
    logits_source = torch.div(logits_source, T)
    logits_target = torch.div(logits_target, T)
    logits_target_soft = F.log_softmax(logits_source, dim=1)
    logits_source_soft = F.softmax(logits_target, dim=1)
    loss_soft = criterion(logits_target_soft, logits_source_soft)
    loss_soft = loss_soft * T * T * 1000
    return loss_soft


def get_feature_loss(features_source, features_target):
    criterion = torch.nn.L1Loss(reduction='mean')
    loss_L1 = criterion(features_source, features_target)

    return loss_L1


def hinge_loss(features=[], targets=[]):
    mat_data = sio.loadmat('LightenedCNN_C_lfw.mat')
    features_of_person = torch.tensor(mat_data.get('features'))

    margin = 2.0
    features = torch.randn(len(features_of_person), 256)
    features = features.div(
        torch.norm(features_of_person, p=2, dim=1, keepdim=True).expand_as(features_of_person))
    targets = torch.randn(1, len(features_of_person)).squeeze(0)
    distances = torch.zeros(len(targets), len(targets))
    for i in range(len(features[0])):
        for j in range(i, len(features[1])):
            distances[i][j] = torch.sum(torch.pow(features[i] - features[j], 2), dim=0)

    print(torch.sum(distances) / torch.sum(torch.tensor(range(len(features_of_person[0])))))
    return 0


hinge_loss()
