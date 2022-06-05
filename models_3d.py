import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # print(out.shape)
        # print(self.shortcut(x).shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=(2, 2, 2, 2), in_planes=8):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Sequential(
            # the first layer
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.layer1 = self._make_layer(block, self.in_planes, num_block[0], stride=1)             # four layers 2-5
        self.layer2 = self._make_layer(block, self.in_planes * 2, num_block[1], stride=2)            # four layers 6-9
        self.layer3 = self._make_layer(block, self.in_planes * 2, num_block[2], stride=2)            # four layers 10-13
        self.layer4 = self._make_layer(block, self.in_planes * 2, num_block[3], stride=2)            # four layers 14-17

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class AdaGN(nn.Module):
    def __init__(self, input_size, channels):
        super(AdaGN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, channels * 2),
            nn.LeakyReLU()
        )

    def forward(self, x, G, factor):
        style = self.fc(factor)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)

        N, C, H, W = x.shape
        x = x.view(N * G, -1)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        x = (x - mean) / (var + 1e-8).sqrt()
        x = x.view([N, C, H, W])

        x = x * (style[:, 0, :, :, :] + 1.) + style[:, 1, :, :, :]
        return x


class SELayer(nn.Module):
    def __init__(self, channel_num, compress_rate):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel_num, (channel_num) // compress_rate, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_num) // compress_rate, channel_num, bias=True),
            nn.Sigmoid()
        )

    def forward(self, feature):
        batch_size, num_channels, H, W = feature.size()
        squeeze_tensor = self.gap(feature)
        squeeze_tensor = squeeze_tensor.view(squeeze_tensor.size(0), -1)
        fc_out = self.se(squeeze_tensor)
        output_tensor = torch.mul(feature, fc_out.view(batch_size, num_channels, 1, 1))
        return output_tensor


class EyeStream(nn.Module):
    def __init__(self):
        super(EyeStream, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.features1_1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(3, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.features1_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SELayer(48, 16),
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.features1_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.features2_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.features2_3 = nn.ReLU(inplace=True)

        self.AGN1_1 = AdaGN(128, 48)
        self.AGN1_2 = AdaGN(128, 64)
        self.AGN2_1 = AdaGN(128, 128)
        self.AGN2_2 = AdaGN(128, 64)

    def forward(self, x, factor):
        x1 = self.features1_3(self.AGN1_2(self.features1_2(self.AGN1_1(self.features1_1(x), 6, factor)), 8, factor))
        x2 = self.features2_3(self.AGN2_2(self.features2_2(self.AGN2_1(self.features2_1(x1), 16, factor)), 8, factor))

        return torch.cat((x1, x2), 1)


class FaceStream(nn.Module):
    def __init__(self, res=False):
        super(FaceStream, self).__init__()
        if res:
            self.conv = ResNet()
            self.fc = nn.Sequential(
                nn.Linear(14 * 14 * 64, 128),
                nn.LeakyReLU(inplace=True),
                nn.Linear(128, 64),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
                nn.GroupNorm(6, 48),
                nn.ReLU(inplace=True),

                nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
                nn.GroupNorm(12, 96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(16, 128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 192),
                nn.ReLU(inplace=True),
                SELayer(192, 16),

                nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
                nn.GroupNorm(16, 128),
                nn.ReLU(inplace=True),
                SELayer(128, 16),

                nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                SELayer(64, 16),
            )
            self.fc = nn.Sequential(
                nn.Linear(5 * 5 * 64, 128),
                nn.LeakyReLU(inplace=True),
                nn.Linear(128, 64),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AFFNet(nn.Module):
    def __init__(self, res=False):
        super(AFFNet, self).__init__()
        self.eyeModel = EyeStream()
        self.eyesMerge_1 = nn.Sequential(
            SELayer(256, 16),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.eyesMerge_AGN = AdaGN(128, 64)
        self.eyesMerge_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(64, 16)
        )
        self.faceModel = FaceStream(res)
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
        )
        # Head Rotation vector
        self.vector_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 96),
            nn.LeakyReLU(inplace=True),
            nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )
        # Gaze Regression
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 3),
        )


    def forward(self, left_eyes, right_eyes, faces, vector):
        face = self.faceModel(faces)
        vector = self.vector_fc(vector)
        factor = torch.cat((face, vector), 1)

        eye_l = self.eyeModel(left_eyes, factor)
        eye_r = self.eyeModel(right_eyes, factor)

        # Attention, AdaGN, FC
        eyes = torch.cat((eye_l, eye_r), 1)
        eyes = self.eyesMerge_2(self.eyesMerge_AGN(self.eyesMerge_1(eyes), 8, factor))
        eyes = eyes.view(eyes.size(0), -1)
        eyes = self.eyesFC(eyes)

        # Cat all
        x = torch.cat((eyes, face, vector), 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    m = AFFNet()
    feature = {"faceImg": torch.zeros(10, 3, 224, 224), "leftEyeImg": torch.zeros(10, 1, 112, 112),
               "rightEyeImg": torch.zeros(10, 1, 112, 112), "vector": torch.zeros(10, 3),
               "label": torch.zeros(10, 3), }
    a = m(feature["leftEyeImg"], feature["rightEyeImg"], feature["faceImg"], feature["vector"])

    face = torch.ones(10, 3, 224, 224)
    print(a.shape)
