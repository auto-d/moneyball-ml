import numpy as nbp 
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

class WARNet(nn.Module): 

    def __init__(self, n_input, n_hidden1, n_hidden2=0): 
        """
        Constructor for our WAR regression network 
        """
        super().__init__()

        # TODO: calculate the input size and test this
        self.input_size = n_input * 2
        self.output_size = 1
        self.hidden = nn.Linear(self.input_size, n_hidden1)

        if n_hidden2: 
            self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
            self.out = nn.Linear(n_hidden2, 1)
        else: 
            self.hidden2 = None
            self.out = nn.Linear(n_hidden1, 1)

    def forward(self, x): 
        """
        Forward pass for our network
        """
        x = self.hidden1(x) 
        x = F.relu(x)
        
        if self.hidden is not None: 
            x = self.hidden2(x)
            x = F.relu(x) 
        
        x = self.out(x) 
        return x