import os

import torch

from device import device
from loss import compute_center_loss, get_center_delta, get_distillation_loss, get_feature_loss
from utils import plot_loss


class Trainer(object):

    # test
    def __init__(
            self, optimizer, scheduler, model1, model2,
            labeled_dataloader, unlabeled_dataloader, test_dataloader,
            log_dir=False, max_epoch=100, resume=False,
            persist_stride=20, lamda=0.03, alpha=0.5, sigma=100, phi=1000):

        self.log_dir = log_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model1 = model1
        self.model2 = model2
        self.max_epoch = max_epoch
        self.resume = resume
        self.persist_stride = persist_stride
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.training_losses = {
            'self_loss': [], 'L1 loss': [],
            'distillation loss': [], 'together': []}
        self.validation_losses = {
            'L1 loss': [], 'distillation loss': [],
            'together': [], 'top3acc': [], 'top1acc': []}
        self.start_epoch = 1
        self.current_epoch = 1
        self.lamda = lamda
        self.alpha = alpha
        self.sigma = sigma
        self.phi = phi
        self.test_dataloader = test_dataloader

        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)


    def train(self):
        for self.current_epoch in range(self.start_epoch, self.max_epoch + 1):
            self.run_epoch(mode='train')
            if not (self.current_epoch % self.persist_stride):  # save models every 'stride' epochs
                self.persist()

    def run_epoch(self, mode):
        if mode == 'train':
            unlabeled_dataloader = self.unlabeled_dataloader
            labeled_dataloader = self.labeled_dataloader
            dataloader_iterator = iter(unlabeled_dataloader)

            loss_recorder = self.training_losses
            self.model2.train()

        self_loss = 0
        total_L1_loss = 0
        total_self_loss = 0
        total_distillation_loss = 0
        total_loss = 0
        batch = 0

        with torch.set_grad_enabled(mode == 'train'):
            for labeled_image, targets in labeled_dataloader:
                try:
                    _, unlabeled_image_source, unlabeled_image_target = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(unlabeled_dataloader)
                    _, unlabeled_image_source, unlabeled_image_target = next(dataloader_iterator)

                batch += 1
                targets = torch.tensor(targets).to(device)
                unlabeled_image_target = unlabeled_image_target.to(device)
                unlabeled_image_source = unlabeled_image_source.to(device)
                labeled_image = labeled_image.to(device)
                centers = self.model2.centers

                # get unlabeled features
                logits_source, features1_unlabeled = self.model1(unlabeled_image_source)
                logits_target, features2_unlabeled = self.model2(unlabeled_image_target)

                # get labeled features and logits
                logits_labeled, features2_labeled = self.model2(labeled_image)

                # compute loss for target domain
                cross_entropy_loss = torch.nn.functional.cross_entropy(
                    logits_labeled, targets)
                center_loss = compute_center_loss(features2_labeled, centers, targets)

                # compute cross-domain loss
                distillation_loss = get_distillation_loss(
                    logits_source=logits_source,
                    logits_target=logits_target
                )

                L1_loss = get_feature_loss(
                    features_source=features1_unlabeled,
                    features_target=features2_unlabeled
                )

                self_loss = self.lamda * center_loss + cross_entropy_loss
                loss = self_loss + L1_loss * self.sigma + distillation_loss * self.phi

                total_self_loss += self_loss
                total_distillation_loss += distillation_loss
                total_L1_loss += L1_loss
                total_loss += loss

                print(
                    "[{}:{}] :  self_loss: {:.8f}\tL1_loss: {:.8f}\tdistillation_loss: {:.8f}\t"
                    "sum_loss: {:.8f}".format(
                        mode, self.current_epoch, self_loss, L1_loss, distillation_loss,
                        loss
                    )
                )

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    center_deltas = get_center_delta(
                        features2_labeled.data, centers, targets, self.alpha)
                    self.model2.centers = centers - center_deltas

            self.scheduler.step()
            print("lr:", self.scheduler.get_last_lr())
            loss_recorder['self_loss'].append(total_self_loss)
            loss_recorder['L1_loss'].append(total_L1_loss)
            loss_recorder['total_distillation_loss'].append(total_distillation_loss)
            loss_recorder['together'].append(total_loss)

            if not (self.current_epoch % 20):
                plot_loss(self.current_epoch, self.log_dir, loss_recorder)
            print(
                "[{}:{}] finished.  total_self_loss: {:.8f}\t total_L1_loss: {:.8f}\t"
                "total_distillation_loss: {:.8f}\t total_loss: {:.8f}".format(
                    mode, self.current_epoch, total_self_loss, total_L1_loss,
                    total_distillation_loss, total_loss
                )
            )

    def _get_matches(self, targets, logits, n=1):
        # get the rank n's index of the line, which is the predicted class
        _, preds = logits.topk(n, dim=1)
        # reshape targets(256) to targets_repeated(256,1) to match preds
        targets_repeated = targets.view(-1, 1).repeat(1, n)
        matches = torch.sum(preds == targets_repeated, dim=1) \
            .nonzero().size()[0]
        return matches

    def persist(self, is_best=False):
        model_dir = os.path.join(self.log_dir, 'models')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        file_name = (
            "epoch_{}_best.pth.tar" if is_best else "epoch_{}.pth.tar") \
            .format(self.current_epoch)

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model2.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        state_path = os.path.join(model_dir, file_name)
        torch.save(state, state_path)
