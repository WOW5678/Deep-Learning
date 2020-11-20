# -*- coding:utf-8 -*-
'''
Create time: 2020/11/18 14:36
@Author: 大丫头
'''
from os import walk
import torch
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
#from torchsummary import summary
from torch.utils.data import Dataset,DataLoader,random_split
import random
import dataset
import models
import torch.nn as nn

def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(r'The model architecture:\n\n', model)
    print(r'\nThe model has {temp:,} trainable parameters')


# saving and loading checkpoint mechanisms
def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path == None:
        return
    save_path = save_path
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

    print(r'Model saved to ==> {save_path}')


def load_checkpoint(model, optimizer):
    save_path = r'siameseNet-batchnorm50.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(r'Model loaded from <== {save_path}')

    return val_loss


# training and validation after every epoch
def train(model, train_loader, val_loader, num_epochs, criterion, save_name):
    best_val_loss = float("Inf")
    train_losses = []
    val_losses = []
    cur_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch + 1))
        for img1, img2, labels in train_loader:
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_running_loss = 0.0
        with torch.no_grad():
            model.eval()
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
              .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(save_name, model, optimizer, best_val_loss)

    print("Finished Training")
    return train_losses, val_losses
# evaluation metrics
def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        print('Starting Iteration')
        count = 0
        for mainImg, imgSets, label in test_loader:
            mainImg = mainImg.to(device)
            predVal = 0
            pred = -1
            for i, testImg in enumerate(imgSets):
                testImg = testImg.to(device)
                output = model(mainImg, testImg)
                if output > predVal:
                    pred = i
                    predVal = output
            label = label.to(device)
            if pred == label:
                correct += 1
            count += 1
            if count % 20 == 0:
                print("Current Count is: {}".format(count))
                print('Accuracy on n way: {}'.format(correct/count))

if __name__ == '__main__':
    # setting the root directories and categories of the images
    root_dir = 'images_background/images_background/'
    #root_dir = 'images_evaluation/images_evaluation/'
    categories = [[folder, os.listdir(root_dir + folder)] for folder in os.listdir(root_dir) if
                  not folder.startswith('.')]

    # choose a training dataset size and further divide it into train and validation set 80:20
    dataSize = 10000  # self-defined dataset size
    TRAIN_PCT = 0.8  # percentage of entire dataset for training
    train_size = int(dataSize * TRAIN_PCT)
    val_size = dataSize - train_size

    transformations = transforms.Compose([transforms.ToTensor()])

    omniglotDataset = dataset.OmniglotDataset(categories, root_dir, dataSize, transformations)
    train_set, val_set = random_split(omniglotDataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=16, shuffle=True)

    # create the test set for final testing
    testSize = 5000
    numWay = 20
    test_root_dir = 'images_evaluation/images_evaluation/'
    test_categories = [[folder, os.listdir(test_root_dir + folder)] for folder in os.listdir(test_root_dir) if
                  not folder.startswith('.')]

    test_set = dataset.NWayOneShotEvalSet(test_categories, test_root_dir, testSize, numWay, transformations)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=2, shuffle=True)

    # showing a sample input of a training set
    # count0 = 0
    # count1 = 0
    # for img1, img2, label in train_loader:
    #     print()
    #     if label[0] == 1.0:
    #         print(img1[0])
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(img1[0][0])
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(img2[0][0])
    #         # print(label)
    #         break
        # break

    # showing a sample input of the testing set
    # count = 0
    # for mainImg, imgset, label in test_loader:
    #     # print(len(imgset))
    #     # print(label)
    #     # print(imgset.shape)
    #     if label != 1:
    #         for count, img in enumerate(imgset):
    #             plt.subplot(1, len(imgset) + 1, count + 1)
    #             plt.imshow(img[0][0])
    #             # print(img.shape)
    #         print(mainImg.shape)
    #         plt.subplot(1, len(imgset) + 1, len(imgset) + 1)
    #         plt.imshow(mainImg[0][0])
    #         count += 1
    #         break
    #     # break

    # creating the original network and couting the paramenters of different networks
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    siameseBaseLine = models.Net()
    siameseBaseLine = siameseBaseLine.to(device)
    count_parameters(siameseBaseLine)

    # actual training
    import torch.optim as optim

    optimizer = optim.Adam(siameseBaseLine.parameters(), lr=0.0006)
    num_epochs = 5
    criterion = nn.BCEWithLogitsLoss()
    save_path = 'siameseNet-batchnorm50.pt'
    train_losses, val_losses = train(siameseBaseLine, train_loader, val_loader, num_epochs, criterion, save_path)

    # Evaluation on previously saved models
    import torch.optim as optim

    load_model = models.Net().to(device)
    load_optimizer = optim.Adam(load_model.parameters(), lr=0.0006)

    num_epochs = 10
    eval_every = 1000
    total_step = len(train_loader) * num_epochs
    best_val_loss = load_checkpoint(load_model, load_optimizer)

    print(best_val_loss)
    eval(load_model, test_loader)

    # # plotting of training and validation loss
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label="Validation Loss")
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.show()

    """How to use the evaluation n way:

    # Set the parameters
    testSize = 5000 # how big you want your test size to be
    numWay = 4 # how many ways metric

    # Create the dataset for it and put it into dataloader
    test_set = NWayOneShotEvalSet(categories, root_dir, testSize, numWay, transformations) 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, num_workers = 2)


    # Load the models (the name of the loaded model can be changed in the load_checkpoint() function)
    load_model = Net().to(device)
    load_optimizer = optim.Adam(load_model.parameters(), lr=0.0006)


    num_epochs = 10
    eval_every = 1000
    total_step = len(train_loader)*num_epochs
    best_val_loss = load_checkpoint(load_model, load_optimizer)

    print(best_val_loss)

    # Evaluate from the test loader 

    eval(load_model, test_loader)

    """