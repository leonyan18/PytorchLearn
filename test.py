import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST('data/', train=True, transform=transformation,
                                           download=True)
test_dataset = torchvision.datasets.MNIST('data/', train=False, transform=transformation,
                                          download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    img = ((mean * image) + std)
    plt.imshow(image, cmap='gray')


sample_data = next(iter(train_loader))
plot_img(sample_data[0][1])
plot_img(sample_data[0][2])
print("object(s), separator=separator, end=end, file=file, flush=flush")


#   卷积神经网络模型
class Jcnn(nn.Module):
    def __init__(self):
        super(Jcnn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(320, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

    def forward(self, X):
        # X=torch.rand(size=(1,3,28,28),dtype=torch.float32)
        X = self.net(X)


#  训练模型
def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()  # 如果模型中有BN层(Batch Normalization）和 Dropout，
        # 需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。
    if phase == 'validation':
        model.eval()  # .eval()不启用 Batch Normalization 和 Dropout
        volatile = True
        running_loss = 0.0
        running_correct = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer = torch.optm.SGD(model.parameters(), lr=0.01)
            if is_cuda:
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile), Variable(target)
                if phase == 'training':
                    optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
                output = model(data)
                loss = F.nll_loss(output, target)
                running_loss += F.nll_loss(output, target, size_averge=False).data[0]
                preds = output.data.max(dim=1, keepdim=True)[1]  # dim=1按行寻找每一行最大值，返回一个表示最大值位置的保持维度不变的数组，
                running_correct += preds.eq(target.data.view_as(preds)).cpu.sum()  # 使用sum统计相等的个数
                if phase == 'training':
                    loss.backward()
                    optimizer.step()
            loss = running_loss / len(data_loader.dataset)  # 和data有什么区别
            accuracy = 100. * running_correct / len(data_loader.dataset)
            print(
                f'{phase} loss is{loss:{5}.{2}} and{phase} accuracy is{running_correct / len(data_loader.dataset)}{accuracy:{10}.{4}}')
        return loss, accuracy


# 迭代预测
model = Jcnn()
is_cuda = False
if is_cuda:
    model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 20):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

# 绘制训练和测试的损失值
plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
plt.legend()
# 绘制训练和测试的准确率
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='training accuracy')
plt.plot(range(1, len(val_losses) + 1), val_accuracy, 'r', label='validation accuracy')
plt.legend()
