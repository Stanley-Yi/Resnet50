# !/usr/local/bin/python3
# @Time : 2021/6/21 9:28
# @Author : Tianlei.Shi
# @Site :
# @File : train.py
# @Software : PyCharm

import json
import os
import time

import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet50_model import ResNet50
from tqdm import tqdm

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "code/resnet50", "data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)


    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    net = ResNet50()

    # model_weight_path = "exp/2021-06-22-10-39-39_L0.508_A0.852/best_resnet50Net.pth"
    # net.load_state_dict(torch.load(model_weight_path))

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 20
    lowest_loss = 100
    best_acc = 0.0
    localtime = time.strftime("%Y-%m-%d-%I-%M-%S")
    save_path = 'exp/' + localtime
    mkfile(save_path)
    best_save_path = save_path + './{}Net.pth'.format('best_resnet50')
    lowestLoss_path = save_path + './{}Net.pth'.format('low_loss_resnet50')
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():

            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        total_loss = running_loss / train_steps
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, total_loss, val_accurate))

        mn = str(epoch + 1) + '_resnet50'
        final_save_path = save_path + './{}Net.pth'.format(mn)
        torch.save(net.state_dict(), final_save_path)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), best_save_path)

        if total_loss < lowest_loss:
            lowest_loss = total_loss
            torch.save(net.state_dict(), lowestLoss_path)

    os.rename(save_path, save_path + '_L%.3f_A%.3f' %(lowest_loss, best_acc))

    print('Finished Training')
    print('the best accuracy is: {}, and the lowest loss is: {}'.format(best_acc, lowest_loss))



if __name__ == '__main__':
    main()
