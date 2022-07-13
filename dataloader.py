
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import glob

class CustomDataset(Dataset):

    def __init__(self, path,n_classes=10,transform=False):
                 
        self.n_classes = n_classes
        self.transform = transform
        
        # Load file list - glob gives you a list of file paths to the images
        self.filelist = glob.glob(path+'/*.png') 
        
        # load labels.
        labels = np.zeros(len(self.filelist))

        for class_i in range(n_classes):
            files_that_are_of_this_class = ['class'+str(class_i) in x for x in self.filelist]
            labels[ files_that_are_of_this_class ] = class_i

        #the labels need to be converted to torch.LongTensor for multi-class classification
        #see the documentation at https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.labels = torch.LongTensor(labels)

        
    def __len__(self):
       
        return len(self.filelist)


    def __getitem__(self, idx):
        """
        Return image no. "idx" x, and its target y.
        """
        
        img = Image.open(self.filelist[idx])  # Open image no. "idx"
                
        if self.transform:
            img = transforms.RandomRotation(180)( img )
        
        # Transform to tesnor
        x = transforms.ToTensor()( img )
        
        # Fix dimensions (add two more layers for RBG)
        x = x.repeat(3,1,1) ## <--- repeat 

        # set target
        y = self.labels[idx]  
    
        return x, y