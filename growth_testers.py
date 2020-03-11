import csv
import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn


with open('lake_jaw_morph.csv') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)


with open('climate_data.csv', 'r') as f:
    reader = csv.reader(f)
    climate_data = list(reader)

with open('eco_site.csv', 'r') as f:
    reader = csv.reader(f)
    eco_site = list(reader)

with open('Watershed_area.csv') as f:
    reader = csv.reader(f)
    area = list(reader)


raw_examples = np.array(raw_examples)[:, 1:len(raw_examples[0])]



lol1 = np.array(raw_examples[:, 29])
lol2 = np.array(raw_examples[:, 1:29])

raw_examples[:, 1] = lol1
for i in range(len(lol2[0])):
    raw_examples[:, i + 2] = lol2[:, i]

for j in range(len(raw_examples[0])):
    print(str(j) + '   ' + raw_examples[0][j])

placeholder = list()
for row in raw_examples:
    placeholder.append(row.tolist())
raw_examples = placeholder
'''
A = 58 KT.diagonal
B = 55 KT.coupler.link (?)
C = 56 KT.input.link (?)
'''
raw_examples[0].append('KT.angle')
for i in range(1, len(raw_examples)):
    a = float(raw_examples[i][58])
    b = float(raw_examples[i][55])
    c = float(raw_examples[i][56])
    raw_examples[i].append(math.acos(((b**2 + c**2) - a**2)/(2*b*c)))



numbers = ['1','2','3','4','5','6','7','8','9','0','.']
for j in range(len(raw_examples[0]) - 1, 0, -1):
    good = 1
    for i in range(len(raw_examples) - 1, 0, -1):
        if good == 1:
            #if raw_examples[i][j] == 'NA' or raw_examples[i][j] == '' or 'raker' in raw_examples[0][j].lower():
            if not (j == 59
                    or 'Standard' in raw_examples[0][j]):
                for i in range(len(raw_examples)):
                    raw_examples[i].pop(j)
                good = 0
                continue
            else:
                try:
                    raw_examples[i][j] = float("".join([c for c in str(raw_examples[i][j]) if c in numbers]))
                except:
                    raw_examples.pop(i)
                    


class Net2(nn.Module):
    def __init__(self, num_classes=10):
        super(Net2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        in_size=x.size(0)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out



net = Net2()
a = net.fit(raw_examples[:,2:len(raw_examples[0])],np.zeros(len(raw_examples[0]) - 2))