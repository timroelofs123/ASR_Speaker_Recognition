import glob, os
import numpy as np
import pickle
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from rnn import RNN


def load_data(data_names, data_labels):
    """load_data: Method which takes the pickle files and takes first 3 seconds of each recording at standard 20x100 MFCC size. 
    We use padding we the size is lower than 100. 
    This method is not used as we want to truncate into multiple recordings to take full advantage of RNNs.

    Args:
        data_names (list): the list of pickle file names which need to be loaded.
        data_labels (list): contains the ids of the speakers.

    Returns: The data as a list of MFCCs having size 20x100 and the labels corresponding to these recordings
    """
    data = []
    data_labels = data_labels.tolist()
    for i in range(len(data_names)):
        with open(data_names[i] + '.pkl', 'rb') as f:
            _, _, mfcc, _, _ = pickle.load(f)
            if len(mfcc[0]) >= 100:
                data.append(mfcc[:, 0:100])
            else:
                target = torch.zeros(20, 100).float()
                source = torch.from_numpy(mfcc[:]).float()
                target[:, :source.shape[1]] = source
                data.append(target.numpy())
    return data, data_labels


def load_data_truncate(data_names, data_labels):
    """load_data_truncate: Method which takes the pickle files and truncates at standard 20x100 MFCC size. 
    Each recording is split into 3 seconds length and all of them are used in training.
    We use padding we the size is lower than 100.

    Args:
        data_names (list): the list of pickle file names which need to be loaded.
        data_labels (list): contains the ids of the speakers.

    Returns: The data as a list of MFCCs having size 20x100 and the labels corresponding to these recordings
    """
    data = []
    new_labels = []
    for i in range(len(data_names)):
        with open(data_names[i] + '.pkl', 'rb') as f:
            _, _, mfcc, _, _ = pickle.load(f)  # load only the mfcc
            if len(mfcc[0]) >= 100:  # If more than 3 seconds
                hundreds = math.floor(
                    len(mfcc[0]) /
                    100)  # See how many 3 seconds segments we have
                # Go through all the segments and add them to the training list
                for j in range(0, hundreds):
                    lower = j * 100
                    higher = (j + 1) * 100
                    data.append(mfcc[:, lower:higher])
                    new_labels.append(data_labels[i])
                # Check if last segment has more than 40% speech part. If most of it is silence we  ignore it
                if (len(mfcc[0]) / 100 - hundreds) > 0.4:
                    lower = hundreds * 100
                    upper = len(mfcc[0])
                    target = torch.zeros(20, 100)
                    source = torch.from_numpy(mfcc[:, lower:upper])
                    target[:, :source.shape[1]] = source
                    data.append(target.numpy())
                    new_labels.append(data_labels[i])
            else:
                # Use only padding if size is lower than 3 seconds
                target = torch.zeros(20, 100).float()
                source = torch.from_numpy(mfcc[:]).float()
                target[:, :source.shape[1]] = source
                data.append(target.numpy())
                new_labels.append(data_labels[i])
    return data, new_labels


def compute_accuracy(preds, targets):
    """compute_accuracy: Method for computing the accuracy.

    Args:
        preds: The predictions of the model for a batch.
        targets: The targets of the batch.

    Returns: The number of correct predictions alongside how many predictions there were
    """
    preds = preds.argmax(dim=1)
    corrects = (preds == targets)
    return corrects.sum(), len(targets)


def main():

    train_names = []
    train_labels = []
    # Read all the file names and the labels of the speakers for the training set
    with open('train.txt', 'r') as file:
        for line in file:
            speaker = line.split('-')[0]
            speech = line.split('-')[1]
            file_path = os.path.join('./LibriSpeech/dev-clean/', speaker,
                                     speech,
                                     line.split('\n')[0])
            train_names.append(file_path)
            train_labels.append(speaker)
    file.close()

    test_names = []
    test_labels = []
    # Read all the file names and the labels of the speakers for the testing set
    with open('test.txt', 'r') as file:
        for line in file:
            speaker = line.split('-')[0]
            speech = line.split('-')[1]
            file_path = os.path.join('./LibriSpeech/dev-clean/', speaker,
                                     speech,
                                     line.split('\n')[0])
            test_names.append(file_path)
            test_labels.append(speaker)
    file.close()

    # The following lines are used for encoding our speakers into one-hot encodings.
    # One-hot is useful for representation as we do not have a large number of speakers: 40.

    label_encoder = LabelEncoder()
    train_data_labels = label_encoder.fit_transform(train_labels)
    n_classes = len(np.unique(train_data_labels))
    print('Number of Train classes', len(np.unique(train_data_labels)))
    binarize = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    train_data_labels = binarize.fit_transform(train_data_labels)

    label_encoder = LabelEncoder()
    test_data_labels = label_encoder.fit_transform(test_labels)
    n_classes = len(np.unique(test_data_labels))
    print('Number of Test classes', len(np.unique(test_data_labels)))
    binarize = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    test_data_labels = binarize.fit_transform(test_data_labels)

    # Loading the data for training and testing

    train, train_labels = load_data_truncate(train_names, train_data_labels)
    val, val_labels = load_data_truncate(test_names, test_data_labels)

    # Preparing the data for the DataLoader so that it can be used in batches

    train = np.array(train).astype(np.float32)
    val = np.array(val).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)
    val_labels = np.array(val_labels).astype(np.float32)
    train_load = []
    for i in range(0, len(train)):
        train_load.append((train[i], train_labels[i]))

    val_load = []
    for i in range(0, len(val)):
        val_load.append((val[i], val_labels[i]))

    # Data Loader for the train set. Batch Size of 4, shuffled
    # and dropping the samples which do not fit the batch size.
    train_dataset = DataLoader(train_load,
                               batch_size=4,
                               shuffle=True,
                               drop_last=True)

    # Data Loader for the test set.
    val_dataset = DataLoader(val_load)

    # Initialize the RNN.
    model = RNN(input_size=100,
                output_size=n_classes,
                hidden_dim=256,
                n_layers=1)

    # Specifying the hyperparameters for training
    n_epochs = 100
    lr = 0.00001

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training part
    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

    for epoch in range(0, n_epochs):

        model.train()

        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        epoch_loss = []  # Store the losses for all batches of an epoch
        correct_predictions = 0
        total_predictions = 0

        # Iterate through data loader
        for i, (x, y) in enumerate(train_dataset):
            # Reshaping for training
            x = Variable(x.view(-1, 20, 100))
            y = Variable(y)
            output, _ = model(x)  # Obtain predictions
            target = torch.argmax(y, dim=1)
            loss = criterion(output, target)
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly
            epoch_loss.append(loss.item())
            # Compute number of correct predictions and total number of predictions
            correct, predicted = compute_accuracy(output, target)
            correct_predictions += correct
            total_predictions += predicted

        # Every 10th epoch present statistics
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}, Accuracy: {}/{}".format(
                np.average(epoch_loss), correct_predictions.item(),
                total_predictions))
            train_accuracy.append(correct_predictions.item() /
                                  total_predictions)
            train_loss.append(np.average(epoch_loss))

            # Evaluate the model on the test set
            model.eval()
            correct_predictions = 0
            total_predictions = 0
            epoch_val_loss = []

            for i, (x, y) in enumerate(val_dataset):
                x = Variable(x.view(-1, 20, 100))
                y = Variable(y)
                output, _ = model(x)
                target = torch.argmax(y, dim=1)
                loss = criterion(output, target)
                epoch_val_loss.append(loss.item())
                correct, predicted = compute_accuracy(output, target)
                correct_predictions += correct
                total_predictions += predicted
            print("Eval Accuracy: {}/{}".format(correct_predictions.item(),
                                                total_predictions))
            test_accuracy.append(correct_predictions.item() /
                                 total_predictions)
            test_loss.append(np.average(epoch_val_loss))

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    preds = []
    targets = []

    for i, (x, y) in enumerate(val_dataset):
        x = Variable(x.view(-1, 20, 100))
        y = Variable(y)
        output, _ = model(x)
        target = torch.argmax(y, dim=1)
        correct, predicted = compute_accuracy(output, target)
        preds.append(output)
        targets.append(target)
        correct_predictions += correct
        total_predictions += predicted
    print("Final Eval Accuracy: {}/{}".format(correct_predictions.item(),
                                              total_predictions))

    with open('accuracy.pickle', 'wb') as f:
        pickle.dump(train_accuracy, f)
        pickle.dump(test_accuracy, f)
    f.close()

    with open('loss.pickle', 'wb') as f:
        pickle.dump(train_loss, f)
        pickle.dump(test_loss, f)
    f.close()

    with open('preds.pickle', 'wb') as f:
        pickle.dump(preds, f)
        pickle.dump(targets, f)
    f.close()


main()