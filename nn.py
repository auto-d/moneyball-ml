import numpy as nbp 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset 
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

        NOTE: the input/x here is the X passed to fit() in the skl pipeline. the skorch 
        torch.utils.data.Dataset will interpret a pandas dataset conveniently, but passes
        the columns here as named parameters (so a named param for every value in X.columns)

        NOTE: the (first) object returned will be cast to np.ndarray and returned 
        from `predict` by skorch. i.e. this *is* our predict function in the sklearn
        pipeline and the return value must abide conversion to an np array. 
        """
        x = self.hidden1(x) 
        x = F.relu(x)
        
        if self.hidden is not None: 
            x = self.hidden2(x)
            x = F.relu(x) 
        
        x = self.out(x) 
        
        return x
    
class MBDataset(Dataset): 
    """
    Custom pytorch-compatible dataset. Adapted from 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    ## TODO adapt this to our training setup

    def __init__(self, feature_df): 

        self.feature_df = feature_df

    def __len__(self): 
        
        # TODO compute the lenth of ... what? 
        return len(self.img_labels) 
    
    def __getitem__(self, idx): 
                
        
        # TODO figure out what we return here ... a row and a label? in an array of bytes? 
        return row, label 
    
def get_data_loader(batch_size=5): 
    """
    Retrieve a pytorch-style dataloader 
    """
    data = MBDataset()
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return loader

# TODO: remove, mining any residual clues in the process, skorch will handle this for us
def train(loader, net, iterations=2):
    """
    Train the model with the provided dataset
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(iterations):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    
    return "Training complete!"