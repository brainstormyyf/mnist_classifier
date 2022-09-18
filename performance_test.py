import torch
import time
import argparse
from dataset import MnistDataset
from main import Classifier
import pandas

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
    #  test trained neural network on training data

    # load trained model
    the_model = Classifier(args)
    PATH = "trained_model.pth"  # the path of trained model
    the_model.load_state_dict(torch.load(PATH))

    score = 0
    items = 0

    for label, image_data_tensor, target_tensor in mnist_test_dataset:
        answer = the_model.forward(image_data_tensor).detach().numpy()
        if (answer.argmax() == label):
            score += 1
            pass
        items += 1

        pass

    print("得分：", score, "  ", "测试总数：", items, "  ", "识别准确率:", score / items)