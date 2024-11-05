# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.hidden_size = 256

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # b x 16 x 31 x 31
            nn.ReLU(), # b x 16 x 31 x 31
            nn.MaxPool2d(kernel_size=2, stride=2), # b x 16 x 15 x 15
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # b x 32 x 15 x 15
            nn.ReLU(), # b x 32 x 15 x 15
            nn.MaxPool2d(kernel_size=2, stride=2), # b x 32 x 7 x 7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # b x 64 x 7 x 7
            nn.ReLU(), # b x 64 x 7 x 7
            nn.MaxPool2d(kernel_size=2, stride=2), # b x 64 x 3 x 3
            nn.Flatten(), # b x 576
            nn.Linear(576, self.hidden_size), # b x 256
            nn.ReLU(), # b x 256
            nn.Linear(self.hidden_size, out_size) # b x out_size
        )

        for layer in self.model:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        return self.model(x.view(-1, 3, 31, 31))

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """

        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    in_size = train_set.shape[1]
    out_size = train_labels.max().item() + 1
    lrate = 0.001
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(lrate, loss_fn, in_size, out_size)

    train_dataset = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for sample in train_loader:
            x = sample['features']
            y = sample['labels']
            total_loss += net.step(x, y)
        losses.append(total_loss)

    predictions = net(dev_set).argmax(dim=1).numpy()

    print(losses)

    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    return losses, predictions, net
