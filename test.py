# !/usr/local/bin/python3
# @Time : 2021/6/21 9:47
# @Author : Tianlei.Shi
# @Site :
# @File : test.py
# @Software : PyCharm
import os
import random

import torch
from resnet50_model import ResNet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# 预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
image_path = "food-101/images/baklava"

pathDir = os.listdir(image_path)  # 取图片的原始路径
filenumber = len(pathDir)
# print(filenumber)
rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
sample = random.sample(pathDir, picknumber)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = ResNet50()
# load model weights
model_weight_path = "exp/2021-06-22-09-22-03/best_resnet50Net.pth"
model.load_state_dict(torch.load(model_weight_path))

# 关闭 Dropout
model.eval()

for im in sample:
    img = Image.open(os.path.join(image_path, im)).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(class_indict[str(predict_cla)], predict[predict_cla].item())