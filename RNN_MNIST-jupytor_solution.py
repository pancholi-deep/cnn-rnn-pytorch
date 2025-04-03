#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# you have to set mini-batch size as a hyperparameter
# batch size : how many samples per batch to load
batch_size = 32

train_data = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)

test_data = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = False,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)

train_data, valid_data = train_test_split(train_data, test_size=0.2, shuffle=True)
print('# of train data : {}'.format(len(train_data)))
print('# of valid data : {}'.format(len(valid_data)))
print('# of test data : {}'.format(len(test_data)))

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_outputs, sequence_len):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_len, n_outputs)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x, _ = self.lstm(x, (h0, c0))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_outputs, sequence_len):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_len, n_outputs)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x, _ = self.gru(x, h0)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

input_size = 28
sequence_len = 28
num_layers = 2
# num_layers = 1
# num_layers = 3
# num_layers = 4

hidden_size = 64
n_outputs = 10

learning_rate = 0.01
# learning_rate = 0.001  # New learning rate value

model = GRU(input_size, hidden_size, num_layers, n_outputs, sequence_len)
# Change to LSTM
# model = LSTM(input_size, hidden_size, num_layers, n_outputs, sequence_len)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# First change: Using SGD optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Second change: Using RMSprop optimizer
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# Third change: Using Adagrad optimizer
#optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

# Use Adam optimizer with L2 regularization
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

loss_function = nn.CrossEntropyLoss()

import copy

n_epochs = 10

train_loss = [] # train loss per epoch
valid_loss = [] # valid loss per epoch

train_acc = [] # train accuracy per epoch
valid_acc = [] # valid accuracy per epoch

# update following two variables whenever valid accuracy improves
best_acc = 0
best_model = copy.deepcopy(model)

for epoch in range(n_epochs):
    model.train() # set model as training mode(for compute gradient)
    train_total = 0
    train_correct = 0
    epoch_train_loss = 0
    for i, data in enumerate(train_loader):
        # In PyTorch, for every mini-batch during the training phase, we have to explicitly
        # set the gradients to zero before starting to do backpropragation with following code
        optimizer.zero_grad()

        inputs, labels = data[0], data[1]
        outputs = model(inputs.squeeze(1))
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss = loss_function(outputs, labels)
        epoch_train_loss += loss.item()

        loss.backward() # compute gradient
        optimizer.step() # update weight & bias in the model with computed gradient

    train_loss.append(epoch_train_loss/len(train_loader))
    train_acc.append(train_correct/train_total)

    model.eval() # set model as evaluation mode
    with torch.no_grad():# we don't need to compute gradient during the evaluation process
        valid_total = 0
        valid_correct = 0
        epoch_valid_loss = 0
        for data in valid_loader:
            inputs, labels = data[0], data[1]
            outputs = model(inputs.squeeze(1))

            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)
            epoch_valid_loss += loss.item()

        valid_loss.append(epoch_valid_loss/len(valid_loader))
        valid_acc.append(valid_correct / valid_total)

    print('[{}/{}]'.format(epoch+1, n_epochs))
    print('training loss : {:.3f}\t training accuracy : {:.3f}'.format(epoch_train_loss/len(train_loader), train_correct/train_total))
    print('validation loss : {:.3f}\t validation accuracy : {:.3f}'.format(epoch_valid_loss/len(valid_loader), valid_correct/valid_total))

    if valid_correct/valid_total > best_acc:
        print('validation accuracy improved {:.5f} ======> {:.5f}'.format(best_acc, valid_correct/valid_total))
        best_acc = valid_correct/valid_total
        best_model = copy.deepcopy(model)

model.eval()
with torch.no_grad():
    test_total = 0
    test_correct = 0
    for data in test_loader:
        inputs, labels = data[0], data[1]
        outputs = model(inputs.squeeze(1))

        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print('test accuracy : {:.3f}'.format(test_acc))

