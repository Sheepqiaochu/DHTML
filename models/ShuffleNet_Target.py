import torch
import torch.nn as nn
from .base import FaceModel


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # original group : channel -- 0,1 0,2 0,3 0,4
    # after transpose : group : channel -- 1,0 2,0 3,0 4,0
    # shuffle the channels
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:  # mode c
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )  # mode d
        else:
            # left part
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            # right part
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNet_Target(FaceModel):
    IMAGE_SHAPE = (224, 224)
    FEATURE_DIMENSION = 1024

    def __init__(self, n_classes, input_size=224, width_mul=0.5):
        super().__init__(num_classes=n_classes, feature_dim=self.FEATURE_DIMENSION)
        num_groups = 2
        assert input_size % 32 == 0
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never   be called.
        # only used for indexing convenience.
        if width_mul == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, self.FEATURE_DIMENSION]
        elif width_mul == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, self.FEATURE_DIMENSION]
        elif width_mul == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, self.FEATURE_DIMENSION]
        elif width_mul == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, self.FEATURE_DIMENSION * 2]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]

        # input_channel = 1
        self.conv1 = conv_bn(1, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        # conv 7*7
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        # Global pool:picture size from 7*7 to 1*1
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

        # self.extract_feature = nn.Linear(
        #     self.stage_out_channels[-1], self.feature_dim)
        # building classifier
        if self.num_classes:
            self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        features = x.view(-1, self.stage_out_channels[-1])

        # features = self.extract_feature(features)
        logits = self.classifier(features) if self.num_classes else None

        feature_normed = features.div(
            torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))
        return logits, feature_normed
