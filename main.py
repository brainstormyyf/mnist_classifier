import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt

# classifier class


class Classifier(nn.Module):

    def __init__(self, args):
        # initialise parent pytorch class
        super().__init__()
        self.args = args

        # define neural network layers
        if self.args.activation_function == 'Sigmoid':
            activation = nn.Sigmoid()
        elif self.args.activation_function == 'LeakyReLU':
            activation = nn.LeakyReLU(0.02)
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            activation,
            nn.LayerNorm(200),

            nn.Linear(200, 10),
            activation

        )

        # create loss function
        if self.args.loss_function == 'BCELoss':
            self.loss_function = nn.BCELoss()
        elif self.args.loss_function == 'MSELoss':
            self.loss_function = nn.MSELoss()

        # create optimiser, using simple stochastic gradient descent
        if self.args.optimiser == 'SGD':
            self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        elif self.args.optimiser == 'Adam':
            self.optimiser = torch.optim.Adam(self.parameters())

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

        pass

    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)

        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, and update the weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

    pass