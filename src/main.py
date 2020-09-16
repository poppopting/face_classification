import os
import time

import numpy as np
from tqdm.auto import tqdm

import cv2
from PIL import Image
from matplotlib import pyplot as plt

from utils import DataGenerator, face_plot

root_path = os.getcwd()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
driver_path = 'chromedriver.exe'
data_generator = DataGenerator(root_path, face_cascade, driver_path)

data_generator.get_idol_faces('鬼娃恰吉')
data_generator.get_idol_faces('王世堅')
print('OK')

from dataset import ImageFolder
from model import CNN_MODEL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

TRAIN_SIZE = 0.8
BATCH_SIZE = 10
N_EPOCHES = 20
LR_RATE = 0.0003
WEIGHT_DECAY = 0.00001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device:', DEVICE)

transform = transforms.Compose([transforms.Resize((64, 64)), 
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

dataset = ImageFolder(Y0_dir='王世堅', Y1_dir='鬼娃恰吉', transform=transform)
train_len = int(len(dataset) * TRAIN_SIZE)
test_len = len(dataset) - train_len
train, test = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

model = CNN_MODEL(input_dim=3 , num_filters=16, n_blocks=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()#(weight=torch.FloatTensor([1., 3.]).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=15)

for epo in tqdm(range(N_EPOCHES)):
    model.train()
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        outputs = model(imgs)
        
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        torch.cuda.empty_cache()
        
    model.eval()
    with torch.no_grad():
        training_loss = 0
        training_acc = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            outputs = model(imgs)

            training_loss += criterion(outputs, targets).item()
            training_acc += (outputs.argmax(dim=1) == targets).sum().cpu().numpy()
        
        testing_loss = 0
        testing_acc = 0
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            outputs = model(imgs)

            testing_loss += criterion(outputs, targets).item()
            testing_acc += (outputs.argmax(dim=1) == targets).sum().cpu().numpy()
    print_sent = 'Epoch {0:2d}/{1} : training_loss: {2:.4f}, training_acc: {3:.2f}, testing_loss: {4:.4f}, testing_acc: {5:.2f}'
    print(print_sent.format(epo+1, N_EPOCHES, training_loss, training_acc/train_len, testing_loss, testing_acc/test_len))

    

name_dic = {0: ["wang shi jian", (255,0,255)],
            1: ['chia ji', (14,201,255)]}
face_plot(model, 'chi.jpg', name_dic, transform, DEVICE)
