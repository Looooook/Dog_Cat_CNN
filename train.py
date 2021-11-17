import numpy as np
import os
import Preprocess
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torchvision import transforms


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 15, kernel_size=(25, 25))
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), padding='same')  # 32 100 100

        # self.mp1 = torch.nn.MaxPool2d(kernel_size=(4, 4))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 32 50 50

        # self.conv2 = torch.nn.Conv2d(15, 30, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)  # 64 50 50

        # self.mp2 = torch.nn.MaxPool2d(kernel_size=(3, 3))
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 64 25 25

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(2, 2))  # 128 24 24

        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 128 12 12

        # self.linear = torch.nn.Linear(750, 4)
        self.linear1 = torch.nn.Linear(18432, 3072)
        self.linear2 = torch.nn.Linear(3072, 512)
        # self.linear3 = torch.nn.Linear(5000, 1000)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(128, 4)

    def forward(self, x):
        in_size = x.size(0)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3(x))
        x = self.mp3(x)
        # print(x.shape)
        x = x.view(in_size, -1)
        # print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        return torch.sigmoid(x)


net = Model().cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


def train_one_epoch(epoch):
    total_loss = 0
    # num_counter = 1
    for batch_idx, data in enumerate(train_loader, 0):
        # print(data.shape)
        inputs, target = data
        inputs, target = inputs.cuda(), target.cuda()
        output = net(inputs).cuda()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 == 19:
            print('EPOCH:{},BATCH_IDX:{},LOSS:{}'.format(epoch + 1, batch_idx + 1, total_loss / 20))
            # num_counter += 1


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            label_hat = net(data).cuda()
            _, predicted = label_hat.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(predicted, labels)

        print('accuracy:{} %,'.format(correct / total * 100))


# 实例化
if __name__ == '__main__':
    file_path = './DogRaw/'
    output_path = './train_imgs/'

    output_path_1 = './train_img/'
    dog_types = ['博美犬', '哈士奇', '拉布拉多', '比格犬']

    np_images_train, np_labels_train = Preprocess.Image_to_Numpy(file_path=file_path, output_path=output_path,
                                                                 dog_types=dog_types).imgs_to_np()
    # print(np_images_train.shape)
    # np_images_train = transform(np_images_train)
    train_data = TensorDataset(np_images_train, np_labels_train)

    transforms.Normalize(mean=0.01, std=0.01)
    # print(train_data)

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

    np_images_test, np_labels_test = Preprocess.Image_to_Numpy(file_path=file_path, output_path=output_path_1,
                                                               dog_types=dog_types).imgs_to_np()
    # np_images_test= np_images_test.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    test_data = TensorDataset(np_images_test, np_labels_test)
    test_loader = DataLoader(test_data, batch_size=10)
    for epoch in range(30):
        train_one_epoch(epoch)

    test()
