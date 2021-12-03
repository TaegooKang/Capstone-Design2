import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #latent_vector = out
        out = self.linear(out)
        return out #latent_vector


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class ResEncoder(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResEncoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        return out


def ResEncoder18():
    return ResEncoder(BasicBlock, [2,2,2,2])

def ResEncoder34():
    return ResEncoder(BasicBlock, [3,4,6,3])

def ResEncoder50():
    return ResEncoder(Bottleneck, [3,4,6,3])

def ResEncoder101():
    return ResEncoder(Bottleneck, [3,4,23,3])

def ResEncoder152():
    return ResEncoder(Bottleneck, [3,8,36,3])


class Projector(nn.Module):
    def __init__(self, mode='mlp'):
        super(Projector, self).__init__()
        if mode == 'linear':
            self.projector = nn.Linear(512,128)
        elif mode == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(512,2048),
                nn.ReLU(),
                nn.Linear(2048,128)
            )
        
    def forward(self, x):
        out = self.projector(x)
        out = F.normalize(out, dim=1)

        return out


class ResProjector(nn.Module):
    def __init__(self, mode='mlp'):
        super(ResProjector, self).__init__()
        self.resencoder = ResEncoder18()
        self.projector = Projector()
    
    def forward(self, x):
        out = self.resencoder(x)
        out = self.projector(out)

        return out
        

class Classifier(nn.Module):
    def __init__(self, num_classes=1):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(512 , num_classes)
    
    def forward(self, x):
        out = self.linear(x)

        return out


# encoder, classifier 분리
class ResNet_sep(nn.Module):
    def __init__(self, pretrained_encoder=None):
        super(ResNet_sep, self).__init__()
        #self.device = self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = ResEncoder18()#.to(self.device)
        self.classifier = Classifier()#.to(self.device)
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.classifier(out)

        return out
    
    def aug_forward(self, x):
        out = self.encoder(x)
        # out = augmentation(out)
        out = self.classifier(out)

        return out

    def freeze_enc(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def save(self, save_path):
        encoder_path = os.path.join(save_path, 'encoder.pt')
        classifier_path = os.path.join(save_path, 'classifier.pt')
        
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), classifier_path)
    
    def load(self, encoder_path, classifier_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
        self.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))


if __name__ == '__main__':
    
    model = ResNet_sep()
    
    x = torch.randn(2,3,224,224)
    z = model(x)
   
    print(z.shape)
    model.save('/root/ktg/Capstone2/checkpoint/Base')


def add_random_noise(latent_vector, label, gamma, ratio):
    # minor class sample 
    minor = latent_vector[label==1]
    # number of minor sample
    num_minor = minor.size(0)
    # number of augmenated sample
    num_aug = int(ratio * num_minor)

    # generate random gaussian noise with elementwise std of minor samples    
    if minor.size(0) > 1:
        std = torch.std(minor, dim=0)
        noise = torch.randn(num_aug,latent_vector.size(1))
        noise = noise.mul_(std)
        noise = gamma * noise
        # select sample
        aug = minor[torch.randperm(minor.size(0))]
        aug = aug[:num_aug]
        aug = torch.add(aug,noise)
        aug_label = torch.LongTensor([1]*num_aug)
        new_latent_vector = torch.cat([latent_vector, aug], dim=0)
        new_label = torch.cat([label, aug_label], dim=0)

    else:
        new_latent_vector = latent_vector
        new_label = label
    
    index = [i for i in range(new_label.size(0))]
    random.shuffle(index)

    new_latent_vector = new_latent_vector[index]
    new_label = new_label[index]

    return new_latent_vector, new_label

def interpolate(latent_vector, label, ratio, gamma):
    # minor class sample 
    minor = latent_vector[label==1]
    # number of minor sample
    num_minor = minor.size(0)
    # number of augmenated sample
    num_aug = int(ratio * num_minor)

    new_latent_vector = latent_vector
    new_label = label

    if num_minor > 2:
        num_generated = 0
        finished = False
        while not finished:
            n1 = random.randint(0,num_minor-1)
            n2 = random.randint(0,num_minor-1)
            if n1 != n2:
                new_minor = (gamma*minor[n1] + (1-gamma)*minor[n2]).unsqueeze(0)
                new_latent_vector = torch.cat([new_latent_vector, new_minor], dim=0)
                new_label = torch.cat([new_label, torch.LongTensor([1])], dim=0)
                num_generated += 1
                if num_generated == num_aug:
                    finished = True
    
    index = [i for i in range(new_label.size(0))]
    random.shuffle(index)

    new_latent_vector = new_latent_vector[index]
    new_label = new_label[index]
    
    return new_latent_vector, new_label


def extrapolate_intra(latent_vector, label, ratio, gamma):
    minor = latent_vector[label==1]
    # number of minor sample
    num_minor = minor.size(0)
    # number of augmenated sample
    num_aug = int(ratio * num_minor)

    new_latent_vector = latent_vector
    new_label = label

    if num_minor > 2:
        num_generated = 0
        finished = False
        while not finished:
            n1 = random.randint(0,num_minor-1)
            n2 = random.randint(0,num_minor-1)
            if n1 != n2:
                new_minor = minor[n1] + gamma*(minor[n1]-minor[n2])
                new_minor = new_minor.unsqueeze(0)
                new_latent_vector = torch.cat([new_latent_vector, new_minor], dim=0)
                new_label = torch.cat([new_label, torch.LongTensor([1])], dim=0)
                num_generated += 1
                if num_generated == num_aug:
                    finished = True
    

    index = [i for i in range(new_label.size(0))]
    random.shuffle(index)

    new_latent_vector = new_latent_vector[index]
    new_label = new_label[index]
    
    return new_latent_vector, new_label


def extrapolate_inter(latent_vector, label, ratio, gamma):
   
    minor = latent_vector[label==1]
    major = latent_vector[label==0]
    # number of minor sample
    num_minor = minor.size(0)
    num_major = major.size(0)
    # number of augmenated sample
    num_aug = int(ratio * num_minor)

    new_latent_vector = latent_vector
    new_label = label

    if num_minor > 2:
        num_generated = 0
        finished = False
        while not finished:
            n1 = random.randint(0,num_minor-1)
            n2 = random.randint(0,num_major-1)
  
            new_minor = (gamma * minor[n1]+(1 - gamma) * major[n2]).unsqueeze(0)
            new_latent_vector = torch.cat([new_latent_vector, new_minor], dim=0)
            new_label = torch.cat([new_label, torch.LongTensor([1])], dim=0)
            num_generated += 1
            if num_generated == num_aug:
                finished = True

    index = [i for i in range(new_label.size(0))]
    random.shuffle(index)

    new_latent_vector = new_latent_vector[index]
    new_label = new_label[index]
    
    return new_latent_vector, new_label
