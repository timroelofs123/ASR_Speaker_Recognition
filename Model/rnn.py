import torch
from torch import nn


class RNN(nn.Module):
    """
    Class which defines a Recurrent Neural Network.
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        """
        Method which initializes the Recurrent Neural Network with the provided parameters.
        
        input_size: shape of the MFCC
        output_size: number of speakers 
        hidden_dim: number of features for the first Fully Connected layer
        n_layers: depth of Recurrent layer. Used only 1.
        """
        super(RNN, self).__init__()

        # Defining some parameters
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(self.input_size,
                          self.hidden_dim,
                          self.n_layers,
                          batch_first=True,
                          nonlinearity='relu')
        # Fully connected layer
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ 
        Overridden Method which passes the provided MFCC and returns a prediction. 
        """

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out[:, -1, :]
        out = self.leakyRelu(self.fc1(out))
        out = self.bn1(out)
        out = self.leakyRelu(self.fc2(out))
        out = self.bn2(out)
        out = self.leakyRelu(self.fc3(out))
        out = self.bn3(out)
        out = self.leakyRelu(self.fc4(out))
        out = self.bn4(out)
        out = self.fc5(out)
        out = self.sigmoid(out)
        # out = F.softmax(out, dim=1)

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Method which generates the first hidden state of zeros which we'll use in the forward pass
        """

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden