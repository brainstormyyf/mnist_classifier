import time
import argparse
import torch
from main import Classifier
from dataset import MnistDataset

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="mnist_train")
    parser.add_argument('--cpu', default=True, action='store_true', help='use cpu')
    parser.add_argument("--train_data", default='mnist_train.csv', action="store", type=str)
    parser.add_argument("--test_data", default='mnist_test.csv', action="store", type=str)
    parser.add_argument("--model_save_path", default='trained_model.pth', action="store", type=str)
    parser.add_argument("--epoch", default=3, action="store", type=int)
    parser.add_argument("--activation_function", default='Sigmoid', action="store", type=str)   # Sigmoid or LeakyReLU
    parser.add_argument("--loss_function", default='MSELoss', action="store", type=str)    # MSELoss or BCELoss
    parser.add_argument("--optimiser", default='SGD', action="store", type=str)    # SGD or Adam

    args = parser.parse_args()
    # create neural network
    C = Classifier(args)

    # create dataset
    mnist_dataset = MnistDataset(args.train_data)
    # remember to modify the path

    # train network on MNIST data set
    epochs = args.epoch
    for i in range(epochs):
        print('training epoch', i+1, "of", epochs)
        for label, image_data_tensor, target_tensor in mnist_dataset:
            C.train(image_data_tensor, target_tensor)
            pass
        pass

    # plot classifier error
    C.plot_progress()

    # save model
    PATH = args.model_save_path
    torch.save(C.state_dict(), PATH)