import torch
import torch.nn as nn
import torchvision


class Resnet18(nn.Module):
    def __init__(self, part='face'):
        super(Resnet18, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        if part == 'eye':
            conv1_weight = torch.mean(model.conv1.weight, dim=1, keepdim=True).repeat(1, 1, 1, 1)  # 取出从conv1权重并进行平均和拓展
            conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 新的conv1层
            model_dict = model.state_dict()  # 获取整个网络的预训练权重
            model.conv1 = conv1  # 替换原来的conv1
            model_dict['conv1.weight'] = conv1_weight  # 将conv1权重替换为新conv1权重
            model_dict.update(model_dict)  # 更新整个网络的预训练权重
            model.load_state_dict(model_dict)  # 载入新预训练权重
        self.model = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        out = self.model(x)
        return out


class AttentionBranch(nn.Module):
    def __init__(self):
        super(AttentionBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 14 * 14, 256),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EyeStream(nn.Module):
    def __init__(self):
        super(EyeStream, self).__init__()
        self.model = Resnet18(part='eye')
        self.attention = AttentionBranch()
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        weight = self.attention(x)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x * weight
        return x


class FaceStream(nn.Module):
    def __init__(self):
        super(FaceStream, self).__init__()
        self.conv = Resnet18()
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GazeResNet(nn.Module):
    def __init__(self):
        super(GazeResNet, self).__init__()
        self.eyeModel = EyeStream()
        self.faceModel = FaceStream()
        self.vector_fc = nn.Sequential(
            # head rotation vector
            nn.Linear(3, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 96),
            nn.LeakyReLU(inplace=True),
            nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
        )
        # Gaze Regression
        self.fc = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, left_eyes, right_eyes, faces, vector):
        face = self.faceModel(faces)
        vector = self.vector_fc(vector)
        eye_l = self.eyeModel(left_eyes)
        eye_r = self.eyeModel(right_eyes)
        # Cat all
        x = torch.cat((eye_l, eye_r, face, vector), 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    m = GazeResNet()
    feature = {"faceImg": torch.zeros(10, 3, 224, 224), "leftEyeImg": torch.zeros(10, 1, 112, 112),
               "rightEyeImg": torch.zeros(10, 1, 112, 112), "vector": torch.zeros(10, 3),
               "label": torch.zeros(10, 3), }
    a = m(feature["leftEyeImg"], feature["rightEyeImg"], feature["faceImg"], feature["vector"])
    print(a.shape)
