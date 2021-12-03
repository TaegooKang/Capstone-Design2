import argparse
import torch
import torch.nn as nn

import torch.optim as optim
import os

from torchvision import datasets, transforms
from loss import FocalLoss, CBLoss, SupConLoss
from utils import AverageMeter
from dataloader import CXR
from tqdm import tqdm
from net import *
from utils import AverageMeter


class FineTuner:
    def __init__(self, encoder, classifier, train_loader ,test_loader, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=True
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.lr_decay_step_size,
            gamma=self.args.lr_decay_gamma
        )
        self.gamma = 0.5
        self.ratio = 0.1

    def train_epoch(self, epoch):
        self.encoder.eval()
        self.classifier.train()
        Loss = AverageMeter('Loss', 0)

        criterion = nn.BCEWithLogitsLoss()
        #criterion = FocalLoss(0.25,2)
        #criterion = CBLoss(beta=0.9999, num_per_class=[358,1402])
        for img, label in tqdm(self.train_loader, desc='train_steps'):
            img = img.to(self.device)
            self.optimizer.zero_grad()
            label = label.squeeze()
            with torch.no_grad():
                latent_vector = self.encoder(img)
                latent_vector = latent_vector.cpu()
            #latent_vector, label = add_random_noise(latent_vector, label, self.gamma, self.ratio)
            #latent_vector, label = interpolate(latent_vector, label, self.gamma, self.ratio)
            #latent_vector, label = extrapolate_inter(latent_vector, label, self.gamma, self.ratio)
            latent_vector, label = extrapolate_intra(latent_vector, label, self.gamma, self.ratio)
            latent_vector = latent_vector.to(self.device)
            pred = self.classifier(latent_vector)
            label = label.unsqueeze(1).to(self.device)
            loss = criterion(pred, label)
            loss.backward()
            Loss.update(loss.item())
            self.optimizer.step()
        torch.cuda.empty_cache()
        self.scheduler.step()
        print(f'Epoch:{epoch} Loss:{Loss.avg:.4f}')
        return Loss.avg
    
    def save_model(self):
        print('Saving the best model...')
        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)
        torch.save(self.classifier.state_dict(), os.path.join(self.args.save_path, 'classifier.pt'))
    
    def fit(self):
        global_loss = 1e8
        for epoch in range(self.args.epochs):
            loss = self.train_epoch(epoch+1)
            if global_loss > loss:
                self.save_model()
                global_loss = loss


def main(args):
    projector = ResProjector()
    projector.load_state_dict(torch.load('/root/ktg/Capstone2/checkpoint/SupCon/Projector/ResProjector.pt', map_location='cpu'))
    encoder = projector.resencoder
    classifier = Classifier()

    #encoder_path = '/root/ktg/Capstone2/checkpoint/US/encoder.pt'
    #classifier_path = '/root/ktg/Capstone2/checkpoint/Base/classifier.pt'

    #encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    #classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # define transforms
    basic_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize
        ]
    )
    aug_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomCrop((224,224), padding=16),
            transforms.ToTensor(),
            normalize
        ]
    
    )

    #train_cxr = CXR(transform=TwoBranch(basic_transform))
    train_cxr = CXR(transform=basic_transform)
    
    #train_aug_cxr = CXR(transform=aug_transform, class_type='abnormal')
    #train_cxr += train_aug_cxr
    
    #train_cxr.undersampling()
    
    test_cxr = CXR(train=False, transform=basic_transform)
    train_loader = torch.utils.data.DataLoader(
            train_cxr,
            batch_size=64,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
    test_loader = torch.utils.data.DataLoader(
            test_cxr,
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    
    trainer = FineTuner(encoder, classifier, train_loader, test_loader, args)
    trainer.fit()


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr-decay-gamma', type=float, default=0.0001)
parser.add_argument('--lr-decay-step-size', type=int, default=50)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--save-path', type=str, default='none')
parser.add_argument('--loss', type=str, choices=['bce', 'focal', 'cb'] ,default='bce')
parser.add_argument('--sampling', type=str, choices=['none', 'up', 'down'], default='none')

args = parser.parse_args()

if __name__ == '__main__':
    main(args)