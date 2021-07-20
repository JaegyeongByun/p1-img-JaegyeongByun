import torch 
import torch.nn as nn
import torch.nn.functional as f

import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.ReLU(x)

        x = self.conv2(x)
        x = F.ReLU(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)


        x = self.conv3(x)
        x = F.ReLu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.fc(x)

        return x


class AgeGenderModel(nn.Module):
    def __init__(self, img_size, num_classes=6):
        super(AgeGenderModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        if img_size == 227:
            self.fc6 = nn.Linear(in_features=221184, out_features=512)  # 227: 221184, 384: 743424
            self.dropout6 = nn.Dropout(p=0.2)        
        elif img_size == 384:
            self.fc6 = nn.Linear(in_features=743424, out_features=512)  # 227: 221184, 384: 743424
            self.dropout6 = nn.Dropout(p=0.2)
        else:
            raise ValueError('img_size only 227 or 384')

        self.fc7 = nn.Linear(in_features=512, out_features=512)
        self.dropout7 = nn.Dropout(p=0.2)
        
        self.fc8 = nn.Linear(in_features=512, out_features=num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = x.view(x.shape[0], -1)

        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout7(x)

        x = self.fc8(x)
        x = self.softmax(x)

        return x

class CustomResNext(nn.Module):
    def __init__(self, model_arch, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(kernel_size=12)
        n_features = 1536
        self.age_classifier = nn.Sequential(
            nn.Linear(n_features, 3),
            )
        self.gender_classifier = nn.Sequential(
            nn.Linear(n_features, 2),
            )
        self.mask_classifier = nn.Sequential(
            nn.Linear(n_features, 3),
            )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        z = self.mask_classifier(x)
        y = self.gender_classifier(x)
        x = self.age_classifier(x)

        return x, y, z        