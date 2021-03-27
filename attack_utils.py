import numpy as np
import torch

from loss import compute_center_loss


class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type):
        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start
        self.norm_type = norm_type

    def perturb(self, model, x, y, centers):
        if self.steps == 0 or self.radius == 0:
            return x.clone()

        adv_x = x.clone()
        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += (torch.rand_like(x) - 0.5) * self.radius / self.steps
            self._clip_(adv_x, x)

        model.eval()
        for step in range(self.steps):
            adv_x.requires_grad_()
            _y, features = model(adv_x)

            cross_entropy_loss = torch.nn.functional.cross_entropy(
                _y, y)
            center_loss = compute_center_loss(features, centers, y)
            loss = center_loss * 0.03 + cross_entropy_loss
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0], -1) ** 2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0], -1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape(-1, *([1] * (len(x.shape) - 1)))
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                self._clip_(adv_x, x)

        return adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0], -1) ** 2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0], -1).abs().sum(dim=1)
            norm = norm.reshape(-1, *([1] * (len(x.shape) - 1)))
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(-0.5, 0.5)


def membership_inference_attack(model, train_loader, test_loader, cpu=False, samp_T=0):
    model.eval()

    result = []
    softmax = torch.nn.Softmax(dim=1)

    train_cnt = 0
    for iid, (x, y) in enumerate(train_loader):
        if not cpu:
            x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _y = None
            for i in range(samp_T):
                tmp_y = softmax(model(x))
                if _y is None:
                    _y = tmp_y
                else:
                    _y += tmp_y
            _y /= samp_T
        train_cnt += len(y)
        for i in range(len(_y)):
            result.append([_y[i][y[i]].item(), 1])

    test_cnt = 0
    for x, y in test_loader:
        if not cpu:
            x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _y = None
            for i in range(samp_T):
                tmp_y = softmax(model(x))
                if _y is None:
                    _y = tmp_y
                else:
                    _y += tmp_y
            _y /= samp_T
        test_cnt += len(y)
        for i in range(len(_y)):
            result.append([_y[i][y[i]].item(), 0])

    result = np.array(result)
    result = result[result[:, 0].argsort()]
    one = train_cnt
    zero = test_cnt
    best_atk_acc = 0.0
    for i in range(len(result)):
        atk_acc = 0.5 * (one / train_cnt + (test_cnt - zero) / test_cnt)
        best_atk_acc = max(best_atk_acc, atk_acc)
        if result[i][1] == 1:
            one = one - 1
        else:
            zero = zero - 1

    return best_atk_acc
