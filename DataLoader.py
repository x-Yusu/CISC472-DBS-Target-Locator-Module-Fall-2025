""""
DataLoader.py

Author:
Leopold Ehrlich

This file contains the DataLoader class, which loads FMRI data, preprocesses it, and presents it as numpy arrays.
"""

import numpy as np
import os

import nilearn

class DataLoader():

    def __init__(self):
        self.pipeline = [
            self.smooth,
            self.correct_motion,
            self.coregister_vols,
            self.extract_brain
            ]

    def __call__(self, *args):
        return self.load_sample(*args)

    
    def load_sample(self, fmri, mri):
        """
        Loads an FMRI file, runs preprocessing and returns it as a numpy array
        """

        as_nifti = lambda x : nib.Nifti1Image(x, np.eye(4))
        fmri, mri = map(as_nifti, (fmri, mri))

        for op in self.pipeline:
            fmri, mri = op(fmri, mri)

        return (fmri, mri)


    def load_directory(self,path=None):
        """
        Generator method for a directory of FMRI data files
        """

        for i in range(20):
            yield self.load_sample()

    def smooth(self, fmri, mri):
        return (fmri, mri)

    def extract_brain(self, fmri, mri):
        mask = nilearn.masking.compute_brain_mask(mri)
        mri  = mri * mask
        fmri = fmri * mask[..., None]
        
        return (fmri, mri)

    def correct_motion(self, fmri, mri):
        pass

    def coregister_vols(self, fmri, mri):
        pass