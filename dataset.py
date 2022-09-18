import torch
from torch.utils.data import Dataset
import pandas
import matplotlib.pyplot as plt

# dataset class


class MnistDataset(Dataset):

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # image target (label)
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0

        # image data, normalised from 0-255 to 0-1
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

        # return label, image data tensor and target tensor
        return label, image_values, target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()
        pass

    pass


if __name__=='__main__':
    mnist_dataset = MnistDataset(r"C:\Users\YYF\PycharmProjects\mnist_classifer\mnist_train.csv")
    # 记得改数据集地址

    # check data contains images

    mnist_dataset.plot_image(19)   # change 19 to access different image

    # check Dataset class can be accessed by index, returns label, image values and target tensor

    mnist_dataset[100]