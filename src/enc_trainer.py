import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import os

from dataloader import CXR
from loss import FocalLoss, CBLoss, SupConLoss
from utils import AverageMeter

from tqdm import tqdm

class EncTrainer:
    def __init__(self, model, train_loader, test_loader, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = optim.SGD(
            self.model.parameters(),
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

    def train_epoch(self, epoch):
        self.model.train()
        Loss = AverageMeter('Loss', 0)

        criterion = SupConLoss()
        
        for img, label in tqdm(self.train_loader, desc='train_steps'):
            img = torch.cat([img[0], img[1]], dim=0)
            img, label = img.to(self.device), label.to(self.device)
            bsz = label.shape[0]
            self.optimizer.zero_grad()
            features = self.model(img)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, label)
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
        new_path = os.path.join(self.args.save_path, f"{self.args.model_name}.pt")
        torch.save(self.model.state_dict(), new_path)
    
    def fit(self):
        global_loss = 1e8
        for epoch in range(self.args.epochs):
            loss = self.train_epoch(epoch+1)
            if global_loss > loss:
                self.save_model()
                global_loss = loss