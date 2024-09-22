import os
import pdb
import csv
import random
import librosa
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from dataset.dataset import TrainDatset, ValDatset, TestDatset


def get_dataloader():
    trainset = TrainDatset()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    valset = ValDatset()
    val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)

    testset = TestDatset()
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def list_to_csv(input_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'category'])  # Writing header
        for i, item in enumerate(input_list):
            writer.writerow([i, item])


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    pred_lst = []

    with torch.no_grad():
        correct = 0
        total = 0
        for audio in test_loader:
            if len(audio.shape) == 4:
                audio = audio.to(device)
            else:
                audio = audio.to(device).unsqueeze(1)

            outputs = model(audio)
            _, predicted = torch.max(outputs.data, 1)
            pred_lst.append(predicted.item())
            total += audio.size(0)

    return pred_lst


def validation(model, val_loader, max_val_acc=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for audio, label in val_loader:
            if len(audio.shape) == 4:
                audio = audio.to(device)
            else:
                audio = audio.to(device).unsqueeze(1)
            label = label.to(device)

            outputs = model(audio)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        if 100*correct/total > max_val_acc:
            max_val_acc = 100*correct/total
        
        print(f'Current Val Acc: {100*correct/total:4f} Max Val Acc: {max_val_acc:4f}%\n')
    
    return 100 * correct / total


def train(epoch, model, train_loader, val_loader, save_path, scheduler_type='s', num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define the scheduler for tuning learning rate
    if scheduler_type == 's':
        # s stands for step
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_type == 'e':
        # e stands for exponential
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_type == 'c':
        # c stands for cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.00008)
    elif scheduler_type == 'p':
        # p stands for plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    max_val_acc = 0
    max_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        for i, (audio, label) in enumerate(train_loader):
            if len(audio.shape) == 4:
                audio = audio.to(device)
            else:
                audio = audio.to(device).unsqueeze(1)
            label = label.to(device)

            # Forward pass
            outputs = model(audio) 
            loss = criterion(outputs, label.long())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}] | Iter {i}/{len(train_loader)} | Loss: {loss.item()}')

        # validation
        val_acc = validation(model, val_loader, max_val_acc)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_epoch = epoch
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
        
        # Update learning rate
        if scheduler_type == 'p':
            scheduler.step(val_acc)
        else:
            scheduler.step()

    return max_val_acc, max_epoch


class SpectrogramResNet(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramResNet, self).__init__()
        # Load a pre-trained ResNet model (e.g., ResNet18)
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

        # Initialize the parameters with uniform distribution
        self._initialize_weights(resnet)
        
        self.resnet = resnet 
    def forward(self, x):                                                                                                                                                              

        return self.resnet(x)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def parse_config():
    parser = argparse.ArgumentParser()
    # ep20_aug04_lr8_s
    # ep: epoch, aug: augmentation probability, lr: learning rate, s: scheduler type
    parser.add_argument('--exp_id', required=False, type=str, default='ep20_aug04_lr8_s', help='config modifications')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print(f"\nExp_ID: {args.exp_id}\n\n")

    # fix seeds for reproducibility
    seed = 0
    if seed is not None:
        cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    save_path = f'results/{datetime.date.today()}/{args.exp_id}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # model = AudioClassifier(input_size=130, output_size=4)
    model = SpectrogramResNet(num_classes=4)

    # get the config list
    config_lst = args.exp_id.split('_')
    
    num_epochs = int(config_lst[0][2:])
    args.epoch = num_epochs

    os.environ['AUGMENT_PROB'] = config_lst[1][3:]
    LR = 0.0001 * int(config_lst[2][2:])

    train_loader, val_loader, test_loader = get_dataloader()

    max_val_acc, max_epoch = train(args.epoch, model, train_loader, val_loader, save_path,
                        scheduler_type=config_lst[3], num_epochs=num_epochs, learning_rate=LR)

    # load the best model
    model.load_state_dict(torch.load(f'{save_path}/best_model.pth'))
    # rename the model weight
    torch.save(model.state_dict(), f'{save_path}/best_model_{max_val_acc:.3f}_ep{max_epoch}.pth')
    model.eval()
    pred_lst = test(model, test_loader)

    # Save prediction.csv
    list_to_csv(pred_lst, f'{save_path}/prediction.csv')

    