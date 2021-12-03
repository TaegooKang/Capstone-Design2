import argparse
import torch
import torch.nn.functional as F
import pickle

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import ResNet18, ResNet_sep
from dataloader import CXR

import sklearn

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

model = ResNet_sep()
encoder_path = '/root/ktg/Capstone2/checkpoint/CB/encoder.pt'
classifier_path = '/root/ktg/Capstone2/checkpoint/CB/extrapolate-intra/classifier.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load(encoder_path, classifier_path)
model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

# define transforms
basic_transform = transforms.Compose(
    [
            transforms.ToTensor(),
            normalize
    ]
)

test_cxr = CXR(train=False, transform=basic_transform)
test_loader = torch.utils.data.DataLoader(
        test_cxr,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

preds, targets = torch.FloatTensor(), torch.FloatTensor()

for idx, (img, target) in enumerate(test_loader):
    img = img.to(device)
    pred = model(img).cpu()
    torch.cuda.empty_cache()
    preds = torch.cat((preds, pred.data), 0) # pred.data -> cuda out of memory 해결?
    targets = torch.cat((targets, target), 0)

preds = F.sigmoid(preds)

preds = preds.numpy()
targets = targets.numpy()
print(roc_auc_score(targets, preds))




