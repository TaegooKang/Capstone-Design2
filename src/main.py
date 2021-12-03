import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import *
from trainer import *
from enc_trainer import *
from utils import *
from dataloader import *

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr-decay-gamma', type=float, default=0.1)
parser.add_argument('--lr-decay-step-size', type=int, default=50)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--save-path', type=str)
parser.add_argument('--loss', type=str, choices=['bce', 'focal', 'cb'] ,default='bce')
parser.add_argument('--sampling', type=str, choices=['none', 'up', 'down'], default='none')

args = parser.parse_args()

def main(args):
   
    model = ResNet_sep()

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
    train_aug_cxr = CXR(transform=aug_transform, class_type='abnormal')
    train_cxr += train_aug_cxr
    
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
    
    trainer = Trainer(model, train_loader, test_loader, args)
    trainer.fit()

if __name__ == '__main__':
    main(args)