#模型建立
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy
from PIL import Image

from torch.nn import functional as F
from d2l import torch as d2l
device = "cuda" if torch.cuda.is_available() else "cpu"


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes:int=1, planes:int=64, stride:int=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #降维，减少通道数
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        #改变图片大小，不改变通道数
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = norm_layer(planes)
        #升维，增加通道数，增加到4倍
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        #降维，减少通道数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #若是layer2、3、4里的第一个bottleneck，stride=2,第一个卷积层会降采样
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #将residual和out相加，经过relu，得到输出
        out += residual
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(self, block, layers, grayscale, num_classes:int = 10, norm_layer = None):
        """
        convert msg object to json string
        if object can't be converted ,return pretty print string
        :param decoded_msg:
        :param msg_name:
        :param msg:
        :return:
        """
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        # 标准化层默认为BN层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # conv1层
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # conv2层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # conv3层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # conv4层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # conv5层
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #512 X 1 X 1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化：若是卷积层采用kaiming_normal()初始化
        # 若是BN层或GroupNorm层则初始化为weight=1, bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                '''
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
                '''
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #每个Bottleneck最后一层bn的weight=0，可提升准确率
        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.zero_()

    def forward(self, x):
        # conv1
        # [1, 224, 224] -> [64, 112, 112]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2
        # [64, 112, 112] -> [64, 56, 56]
        x = self.maxpool(x)
        # [64, 56, 56] -> [64, 56, 56]
        x = self.layer1(x)
        # conv3
        # [64, 56, 56] -> [128, 28, 28]
        x = self.layer2(x)
        # conv4
        # [128, 28, 28] -> [256, 14, 14]
        x = self.layer3(x)
        # conv5
        # [256, 14, 14] -> [512, 7, 7]
        x = self.layer4(x)
        # MNIST本身是1x1：禁用avgpool
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
    def _make_layer(self, block, planes:int, blocks:int, stride:int = 1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        # 判断stride是否为1， 输入通道和输出通道是否相等。不相等使用conv2d作为降采样downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
#                nn.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 添加第一个Bottleneck， downsample为降采样层
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        # 修改输出通道数
        self.inplanes = planes * block.expansion
        # 继续添加这个layer里的Bottleneck
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer))

        return nn.Sequential(*layers)


def resnet50(num_classes:int = 10):
    """Constructs a ResNet-50 model."""
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   grayscale=True,
                   num_classes=num_classes,
                   )
    print(model)
    return model


# if __name__ == "__main__":
#     resnet50()