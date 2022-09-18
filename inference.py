import torch
import time
import argparse
from dataset import MnistDataset
from main import Classifier
import pandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description="mnist_train")
    parser.add_argument('--cpu', default=True, action='store_true', help='use cpu')
    parser.add_argument("--train_data", default='mnist_train.csv', action="store", type=str)
    parser.add_argument("--test_data", default='mnist_test.csv', action="store", type=str)
    parser.add_argument("--model_save_path", default='trained_model.pth', action="store", type=str)
    parser.add_argument("--epoch", default=3, action="store", type=int)
    parser.add_argument("--activation_function", default='Sigmoid', action="store", type=str)  # Sigmoid or LeakyReLU
    parser.add_argument("--loss_function", default='MSELoss', action="store", type=str)  # MSELoss or BCELoss
    parser.add_argument("--optimiser", default='SGD', action="store", type=str)  # SGD or Adam
    args = parser.parse_args()

    # load MNIST test data
    mnist_test_dataset = MnistDataset(r"C:\Users\YYF\PycharmProjects\mnist_classifer\mnist_test.csv")

    # pick a record
    record = 19

    # plot image and correct label
    mnist_test_dataset.plot_image(record)

    # load trained model
    the_model = Classifier(args)
    PATH = "trained_model.pth"  # the path of trained model
    the_model.load_state_dict(torch.load(PATH))

    # visualise the answer given by the neural network

    image_data = mnist_test_dataset[record][1]

    # query from trained network
    output = the_model.forward(image_data)

    # plot output tensor
    pandas.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1))
    plt.show()