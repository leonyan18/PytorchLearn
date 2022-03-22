import os

from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        # 重新建立索引,train_test_split划分时会打乱索引
        self.images_filepaths = images_filepaths.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, index):
        # TODO 需自己修改
        image_filepath = self.images_filepaths[index]
        image = Image.open(os.path.join('./data/' + image_filepath))
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        # print(image.shape)
        return image, label
