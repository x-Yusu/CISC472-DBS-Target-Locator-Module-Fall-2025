""""
DataLoader.py

Author:
Leopold Ehrlich

This file contains the DataLoader class, which loads FMRI data, preprocesses it, and presents it as numpy arrays.
"""

import numpy as np
import os

class DataLoader():

    def __init__(self):
        pass

    def __call__(self, path=None):
        if os.path.isdir(path):
            return self.load_directory(path)
        
        else:
            return self.load_sample(path)

    
    def load_sample(self,path=None):
        """
        Loads an FMRI file, runs preprocessing and returns it as a numpy array
        """
        return np.random.rand(100,100,100,20)


    def load_directory(self,path=None):
        """
        Generator method for a directory of FMRI data files
        """

        for i in range(20):
            yield self.load_sample()
