import pandas as pd
import timm.optim
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from timm.data.transforms_factory import create_transform
from timm.scheduler.step_lr import StepLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from yan.dataset import MyDataSet
from yan.model import YCXNet


class TrainLoop:
    def __init__(self, config_path):
        file = open(config_path, encoding='utf-8')
        # 初始化参数 不在其他地方初始化 避免混淆
        self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.data_path = self.config["data"]["train_path"]
        self.batch_size = self.config["train"]["batch_size"]
        self.num_epochs = self.config["train"]["num_epochs"]
        self.batch_display = self.config["train"]["batch_display"]
        self.lr = self.config["model"]["optimizer"]["lr"]
        self.momentum = self.config["model"]["optimizer"]["momentum"]
        self.decay_t = self.config["model"]["scheduler"]["decay_t"]
        self.decay_rate = self.config["model"]["scheduler"]["decay_rate"]

    def make_data(self):
        # TODO 划分验证集以及K折交叉
        # trans = create_transform(224)
        # df = pd.read_csv(self.data_path)
        # le = LabelEncoder()
        # df['label'] = le.fit_transform(df['label'])
        # # 划分训练集和测试级
        # x_train, x_test, y_train, y_test = train_test_split(df['image'], df['label'], test_size=0.2, random_state=0)
        # train_dataset = MyDataSet(x_train, y_train, trans)
        # test_dataset = MyDataSet(x_test, y_test, trans)
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST('data/', train=True, transform=transformation, download=True)
        test_dataset = torchvision.datasets.MNIST('data/', train=False, transform=transformation, download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def run(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # 选择自己的模型
        model = YCXNet()
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = F.nll_loss
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        # 使用了timm的optimizer和scheduler
        optimizer = timm.optim.create_optimizer_v2(model, opt='lookahead_adam', lr=self.lr)
        scheduler = StepLRScheduler(optimizer, decay_t=self.decay_t, decay_rate=self.decay_rate)
        model.to(device)
        for epoch in range(self.num_epochs):
            train_loader, test_loader = self.make_data()
            self.train(epoch, self.batch_display, train_loader, device, optimizer, model, criterion, scheduler)
            self.test(test_loader, device, model)
            scheduler.step(epoch + 1)

    def train(self, epoch, batch_display, train_loader, device, optimizer, model, criterion, scheduler):
        running_loss = 0.0
        num_steps_per_epoch = len(train_loader)
        num_updates = epoch * num_steps_per_epoch
        batch_idx = 0
        for data in tqdm(train_loader):
            inputs, target = data
            # TODO windows提示转化错误
            # print(target.dtype)
            target = target.type(torch.LongTensor)
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step_update(num_updates=num_updates)
            running_loss += loss.item()
            if batch_idx % batch_display == batch_display - 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / batch_display))
                running_loss = 0.0
            batch_idx = batch_idx + 1

    def test(self, test_loader, device, model):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, target = data
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print('Accuracy on test set: %.3f %% [%d/%d]' % (100.0 * correct / total, correct, total))
