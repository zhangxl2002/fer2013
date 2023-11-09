from django.shortcuts import render, HttpResponse
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
# a=16
# b=32
# c=120
# d=64
# class LeNet(nn.Module):

#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, a, kernel_size=5)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(a, b, kernel_size=5)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(b * 9 * 9, c)
#         self.relu3 = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc2 = nn.Linear(c, d)
#         self.relu4 = nn.ReLU()
#         self.fc3 = nn.Linear(d, 7)

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, b * 9 * 9)
#         x = self.dropout(x)
#         x = self.relu3(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu4(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

a=16
b=32
c=156 #128
d=100
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, a, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(a, b, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(b * 10 * 10, c)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) #0.4
        self.fc2 = nn.Linear(c, d)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(d, 7)
        # self.fc3 = nn.Linear(d, e)
        # self.relu5 = nn.ReLU()
        # self.fc4 = nn.Linear(e, 7)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, b * 10 * 10)
        x = self.dropout(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.relu4(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.relu5(self.fc3(x))
        # x = self.dropout(x)
        # x = self.fc4(x)
        return x
# Create your views here.
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request, *args, **kwargs):
        image = request.data['file']
        img = Image.open(image)
        img = img.resize((48, 48))
        # 转换为灰度图像
        img = img.convert('L')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 将图像转换为 NumPy 矩阵
        feature = torch.tensor(np.array(img).reshape((1,48,48)),dtype=torch.float32).to(device)/255

        print(device)
        model = LeNet1()
        model.load_state_dict(torch.load("/home/zxl/fer2013/LeNet1_2023-11-09_12-48-26_59.96"))
        model.to(device) 
        model.eval()
        with torch.no_grad():
            prediction = model(feature)
            print(prediction)
            predicted_labels = torch.argmax(prediction, dim=1)[0].item()
        print(predicted_labels)

        return Response(predicted_labels, status=status.HTTP_200_OK)