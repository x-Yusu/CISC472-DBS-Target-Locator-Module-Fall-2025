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
        self.pipeline = [self.extract_brain,self.correct_motion,self.coregister_vols]

    def __call__(self, path=None):
        if os.path.isdir(path):
            return self.load_directory(path)
        
        else:
            return self.load_sample(path)

    
    def load_sample(self, fmri, mri):
        """
        Loads an FMRI file, runs preprocessing and returns it as a numpy array
        """

        for op in self.pipeline:
            fmri, mri = op(fmri, mri)

        return img


    def load_directory(self,path=None):
        """
        Generator method for a directory of FMRI data files
        """

        for i in range(20):
            yield self.load_sample()


    def extract_brain(self, fmri, mri):
        pass

    def correct_motion(self, fmri, mri):
        pass

    def coregister_vols(self, fmri, mri):
        pass