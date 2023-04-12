from flask import Flask
import cv2
#from __future__ import annotations
import numpy as np
from flask import render_template, jsonify, redirect, url_for, request
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
from typing import ClassVar, Optional, List

dic = {0 : 'Normal', 1 : 'Non Severe OA', 2 : 'Severe OA'}

app = Flask(__name__)

# ResNet 18 Model

class ResidualBlock(nn.Module):

  expansion: ClassVar[int] = 1

  def __init__(self, in_channels: int, out_channels: int, stride: Optional['int'] = 1, downsample: Optional['bool'] = None) -> None:
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(out_channels)
    )
    self.downsample = downsample
    self.relu = nn.ReLU()
    self.out_channels = out_channels
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out



class ResNet(nn.Module):
  def __init__(self, block: object, layers: List[int], num_classes: Optional['int'] = 3,  zero_init_residual: Optional[bool] = False) -> None:
    super(ResNet, self).__init__()
    self.inplanes: int = 64
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU() 
    )
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
    self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*block.expansion, 3)
  
  def _make_layer(self, block: object, planes: int, blocks: int, stride: Optional[int] = 1) -> nn.Sequential():
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
          nn.BatchNorm2d(planes * block.expansion)
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    
    return nn.Sequential(*layers)
  
  def forward(self, x: torch.tensor) -> torch.tensor:
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

device = torch.device('cpu')
PATH = "Model Version3.pth"
model = ResNet(ResidualBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load(PATH,map_location=device))
model.eval()

def image_transform(image):
    transform = A.Compose([
            A.Resize(width=299, height=299),
    		A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, p=1.0),
    		A.CLAHE(clip_limit=4.0, tile_grid_size=(9,9), p=1.0),
    		A.CenterCrop(width=280, height=200),
    		A.Resize(width=299, height=299),
    		A.HorizontalFlip(p=0.5),
    		A.Rotate(limit=10, p=0.5),
    		A.Normalize(mean=(0.6093), std=(0.1534)),
    		ToTensorV2()
    ]) 
    return transform(image=image)['image'].unsqueeze(0)
    
def model_predict(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_tensor = image_transform(gray)
    print(img_tensor.shape)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    pred = predicted.numpy()
    class_name = dic[pred[0]]
    return class_name

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET','POST'])
def diagnose_page():
    if request.method == 'POST':
        file = request.files['file']
        img_path = "uploads/" + file.filename
        file.save(img_path)
        p = model_predict(img_path)
        return render_template('diagnosis.html',prediction_text=p)
    return render_template('diagnosis.html')

@app.route('/team')
def team_page():
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)
