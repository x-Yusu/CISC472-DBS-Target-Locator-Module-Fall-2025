""""
DataLoader.py

Author:
Leopold Ehrlich

This file contains the DataLoader class, which loads FMRI data, preprocesses it, and presents it as numpy arrays.
"""

import numpy as np
import os

import nilearn as nil
import nibabel as nib

from nipy.algorithms.registration import HistogramRegistration
from scipy.ndimage import affine_transform

from itertools import repeat

class DataLoader():

    def __init__(self):
        self.pipeline = [
            self.smooth,
            self.coregister_all_vols,
            self.extract_brain
            ]


    def __call__(self, *args):
        return self.load_sample(*args)

    
    def load_sample(self, fmri, mri):
        """
        Loads an FMRI file, runs preprocessing and returns it as a numpy array
        """

        if isinstance(fmri, np.ndarray):
            fmri, mri = map(as_nifti, (fmri, mri))

        elif not isinstance(fmri, nibabel.nifti1.Nifti1Image):
            raise ValueError("Inputs must be of type np.ndarray or nibabel.nifti1.Nifti1Image")

        for op in self.pipeline:
            fmri, mri = op(fmri, mri)

        return (fmri, mri)

    def as_nifti(self, vol):
        return nib.Nifti1Image(vol, np.eye(4))

    def load_directory(self,path):
        """
        Generator method for a directory of FMRI data files
        """
        
        fmri_iter = os.paths.scandir(os.path.join(path,"fmri"))
        mri_iter  = os.paths.scandir(os.path.join(path,"mri"))


        for frmi, mri in zip(fmri_iter, mri_iter):
            fmri = nib.load(fmri.path)
            mri  = nib.load(mri.path)
            
            yield self.load_sample(fmri,mri)


    def smooth(self, fmri, mri):
        smoothing_func = lambda x : nil.image.smooth_img(x)
        fmri, mri = map(as_nifti, (fmri, mri))
        
        return (fmri, mri)


    def extract_brain(self, fmri, mri):
        mask = nil.masking.compute_brain_mask(mri)
        mri  = mri * mask
        fmri = fmri * mask[..., None]

        return (fmri, mri)

    def coregister_all_vols(self, fmri, mri):
        mri_data  = mri.get_fdata()
        fmri_data = fmri.get_fdata()

        # Register the first fMRI frame to the structural mri
        fmri_data[...,0] = register_vols(mri, fmri_data[...,0])

        # Register all fmris to the first fMRI
        for i in range(1, fmri.shape[-1]):
            fmri_data[...,i] = register_vols(fmri_data[...,0], fmri_data[...,i])

        return (fmri_data, mri_data)

    def register_vols(v1,v2, mode='rigid'):
        reg = HistogramRegistration(v1, v2, similarity=similarity)
        T = reg.optimize(mode)

        registered = affine_transform(
            v2,
            np.linalg.inv(T.as_affine()[:3, :3]),
            offset=-T.as_affine()[:3, 3],
            order=3
        )

    def normalize_to_template(self, mri, fmri):
        template = nil.datasets.load_mni152_template()
     