import numpy as np 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skorch import NeuralNetRegressor

class WARNet(nn.Module): 

    def __init__(self, n_input, n_hidden1=1, n_hidden2=0, n_hidden3=0):
        """
        Constructor for our WAR regression network 

        I'm clueless about network geometry, allow variation of depth and height as
        well as whether to apply relu activation between the hidden layers. 
        """
        super().__init__()

        self.linear1 = nn.Linear(n_input, n_hidden1)
        
        if n_hidden2: 
            self.linear2 = nn.Linear(n_hidden1, n_hidden2)

            if n_hidden3: 
                self.linear3 = nn.Linear(n_hidden2, n_hidden3)
                self.out = nn.Linear(n_hidden3, 1)
            else: 
                self.linear3 = None
                self.out = nn.Linear(n_hidden2, 1)    
        else: 
            self.linear2 = None
            self.linear3 = None
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
        x = self.linear1(x) 
        x = F.relu(x)
        
        if self.linear2: 
            x = self.linear2(x) 
            x = F.relu(x)

            if self.linear3: 
                x = self.linear3(x) 
                x = F.relu(x)

        x = self.out(x) 
        
        return x
    
class MBDataset(Dataset): 
    """
    Custom pytorch-compatible dataset. Adapted from 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """
    def __init__(self, feature_df, labels): 
        """
        Initialize our custom pytorch dataset object
        """
        
        self.X = feature_df
        self.y = labels

    def __len__(self): 
        """
        Retrieve the length of the dataset
        """
        
        return len(self.X) 
    
    def __getitem__(self, idx): 
        """
        Retrieve the item at the provided index!

        NOTE: a dataset resulting from repeated calls to this fucntion will 
        be provided to our forward pass! this signature should match what's expected there
        """
        row = self.X.iloc[idx]
        X = row.to_numpy().astype(np.float32)
        y = self.y.iloc[idx]
        
        return X, y
    
def get_data_loader(feature_df, labels, batch_size=5): 
    """
    Retrieve a pytorch-style dataloader ... not actually necessary with skorch 
    """
    data = MBDataset(feature_df, labels)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return loader 