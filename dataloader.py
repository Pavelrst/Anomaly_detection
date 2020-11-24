import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import random
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class OneClassDatasetCIFAR10(Dataset):
    """One class dataset prepared from CIFAR10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html"""

    def __init__(self, root_dir, real_class=1, train=True, vis=False):
        self.root_dir = root_dir
        self.real_class = real_class
        self.samples = []
        self.train = train
        self.vis = vis
        for file_name in os.listdir(self.root_dir):
            if self.train:
                if 'data_batch' in file_name:
                    data_dict = unpickle(os.path.join(self.root_dir, file_name))
                    labels = data_dict[b'labels']
                    datas = data_dict[b'data']
                    self.samples += [(data, label) for data, label in zip(datas, labels) if label == self.real_class]
                    print(file_name, " loaded")
            else:
                if 'test_batch' in file_name:
                    data_dict = unpickle(os.path.join(self.root_dir, file_name))
                    labels = data_dict[b'labels']
                    datas = data_dict[b'data']
                    self.samples += [(data, label) for data, label in zip(datas, labels) if label == self.real_class]
                    print(file_name, " loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.samples[idx][0]
        image = data.reshape((3, 32, 32))

        good_shape = image.shape

        label = random.randint(0, 3)
        image = np.rot90(image, k=label, axes=(1, 2)).copy()
        image = image / 255

        assert good_shape == image.shape

        if self.vis:
            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.title('label:'+str(label))
            plt.show()

        return image, label