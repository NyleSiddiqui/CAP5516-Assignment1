import torch
import os
import numpy as np
import torchvision

from tqdm import tqdm
import torch.nn as nn
from torchvision.models import resnet18, resnet101, resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torch.utils.tensorboard import SummaryWriter

from dataloader import CustomDataLoader

def get_transforms(train):
  t = [transforms.Resize([256, 256])]
  if train:
    t.append(transforms.RandomCrop((224, 224)))
    t.append(transforms.RandomHorizontalFlip(0.5))
    t.append(transforms.RandomVerticalFlip(0.5))
    # t.append(transforms.GaussianBlur((7, 7), 2))
  else:
    t.append(transforms.CenterCrop((224, 224)))
  t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
  return transforms.Compose(t)

def train_epoch(model, loader, optimizer, batch_size, writer, epoch):
  model.train()
  correct = 0
  for (img, label) in tqdm(train_loader):
    img, label = img.cuda(), label.cuda()
    pred = model(img)
    loss = nn.CrossEntropyLoss()(pred, label)
    writer.add_scalar("Loss/train", loss, epoch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = torch.argmax(pred, dim=1)
    correct += (pred == label).float().sum()
  accuracy = correct / (len(loader) * batch_size)
  print(f'Train Accuracy: {accuracy}')


def val_epoch(model, loader, batch_size, writer, epoch):
  model.eval()
  correct = 0
  with torch.no_grad():
    for (img, label) in tqdm(test_loader):
      img, label = img.cuda(), label.cuda()
      pred = model(img)
      pred = torch.argmax(pred, dim=1)
      correct += (pred == label).float().sum()
  accuracy = correct / (len(loader) * batch_size)
  print(f'Test Accuracy: {accuracy}')
  writer.add_scalar("Acc/test", accuracy, epoch)
  return accuracy


if __name__ == '__main__':
    num_epochs = 50
    val_epoch_int = 5
    batch_size = 128
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    count = 0
    for child in model.children():
        count += 1
        if count == 7:
            break
        for param in child.parameters():
            param.requires_grad = False
    model.cuda()
    train_transforms = get_transforms('train')
    test_transforms = get_transforms('test')
    writer = SummaryWriter()
    
    train_data = CustomDataLoader('train', train_transforms)
    test_data = CustomDataLoader('test', test_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    max_acc = 0
    for i in range(num_epochs):
        train_epoch(model, train_loader, optimizer, batch_size, writer, i)
        acc = val_epoch(model, test_loader, batch_size, writer, i)
        max_acc = acc if acc > max_acc else max_acc
        print('max acc: ', max_acc.item())
    print('final max acc: ', max_acc.item())
    writer.close()