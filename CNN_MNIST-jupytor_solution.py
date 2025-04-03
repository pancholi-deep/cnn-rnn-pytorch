#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
#from torchsummary import summary

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


class CNN_1(nn.Module):
    def __init__(self, n_channel, n_outputs, conv_kernel_size):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channel,
                               out_channels=6,
                               kernel_size=conv_kernel_size,
                               stride=1,
                               padding='same')
        # init.xavier_uniform_(self.conv1.weight)  # Xavier weight initialization
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool2 = nn.AvgPool2d(2, 2)  # Changed to AvgPool2d

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=conv_kernel_size,
                               stride=1,
                               padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)
        # self.dropout = nn.Dropout(dropout_rate)  # Add dropout layer
    
    def forward(self, x):
        x = F.relu(self.conv1(x)) # conv1 -> ReLU
        # x = F.sigmoid(self.conv1(x))  # Change F.relu to F.sigmoid
        # x = F.tanh(self.conv1(x))  # Change F.relu to F.tanh
        # x = F.leaky_relu(self.conv1(x))  # Change F.relu to F.leaky_relu


        x = self.pool1(x)
        x = F.relu(self.conv2(x)) # conv2 -> ReLU
        x = self.pool2(x)
        x = x.view(-1, 16*7*7) # flatten
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x

batch_size = 32

n_epochs = 10
learning_rate = 0.01
model = CNN_1(1, 10, 8)

# Changed kernel size to 5
# model = CNN_1(1, 10, 5)

# Changed kernel size to 3
# model = CNN_1(1, 10, 3)

# Changed kernel size to 7
# model = CNN_1(1, 10, 7)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# Change optimization method to AdaGrad
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
# Change optimization method to RMSProp
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# Change optimization method to AdaDelta
# optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# Set weight_decay for L2 regularization in Adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


loss_function = nn.CrossEntropyLoss()


train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)


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
        outputs = model(inputs)
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
            outputs = model(inputs)

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


import matplotlib.pyplot as plt
import numpy as np

np_loss = np.array(valid_loss)
loss_ind_sorted = np.argsort(np_loss)
loss_min_ind = loss_ind_sorted[0]

np_acc = np.array(valid_acc)
acc_ind_sorted = np.argsort(np_acc)
acc_max_ind = acc_ind_sorted[-1]

x = [i+1 for i in range(len(train_loss))]


plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x, train_loss, label='train_loss')
plt.plot(x, valid_loss, label='valid_loss')
plt.axvline(loss_min_ind+1, color='red')
plt.legend()

plt.subplot(1,2,2)
plt.title('Accuracy curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc, label='train_acc')
plt.plot(x, valid_acc, label='valid_acc')
plt.axvline(acc_max_ind+1, color='red')
plt.ylim(0.8,1)
plt.legend()


model.eval()
with torch.no_grad():
    test_total = 0
    test_correct = 0
    for data in test_loader:
        inputs, labels = data[0], data[1]
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print('test accuracy : {:.3f}'.format(test_acc))
