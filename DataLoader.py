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
            self.coregister_all_vols,
            self.normalize_to_template,
            self.as_nifti,
            self.smooth,
            self.extract_brain
            ]


    def __call__(self, *args):
        return self.load_sample(*args)

    
    def load_sample(self, fmri, mri):
        """
        Loads an FMRI file, runs preprocessing and returns it as a numpy array
        """

        for op in self.pipeline:
            fmri, mri = op(fmri, mri)

        return (fmri, mri)


    def load_directory(self,path):
        """
        Generator method for a directory of FMRI data files
        """
        
        fmri_iter = os.scandir(os.path.join(path,"fmri"))
        mri_iter  = os.scandir(os.path.join(path,"mri"))


        for fmri, mri in zip(fmri_iter, mri_iter):
            fmri = nib.load(fmri.path)
            mri  = nib.load(mri.path)
            
            yield self.load_sample(fmri,mri)
 

    def coregister_all_vols(self, fmri, mri):
        mri_data  = mri.get_fdata()
        fmri_data = fmri.get_fdata()

        # Register the first fMRI frame to the structural mri
        fmri_data[...,0] = register_vols(mri, fmri_data[...,0])

        # Register all fmris to the first fMRI
        for i in range(1, fmri.shape[-1]):
            fmri_data[...,i] = register_vols(fmri_data[...,0], fmri_data[...,i])

        return (fmri_data, mri_data)


    def register_vols(self, v1, v2, mode='rigid', return_transform=False):
        reg = HistogramRegistration(v1, v2)
        T = reg.optimize(mode)

        v2 = self.apply_affine(v2, T)

        if return_transform: return (v2, T)
        return v2


    def apply_affine(self, target, transform):
        target = affine_transform(
            target,
            np.linalg.inv(transform.as_affine()[:3, :3]),
            offset=-transform.as_affine()[:3, 3],
            order=3
        )

        return target


    def normalize_to_template(self, fmri, mri):
        """
        Does an affine transform to map fmri to a template
        """

        template = nil.datasets.load_mni152_template()
        mri, mri_t = self.register_vols(template, mri, 'affine', True)

        # Find transform for one fMri
        fmri[...,0], fmri_t = self.register_vols(template, fmri[...,0], 'affine', True)

        # Register all fmris to the first fMRI
        for i in range(1, fmri.shape[-1]):
            fmri[...,i] = apply_affine(fmri[...,i], fmri_t)
            
        self.mri_t  = mri_t
        self.fmri_t = fmri_t

        return (fmri,mri)


    def as_nifti(self, fmri, mri):
        """
        Casts the pair of fmri and mri numpy arrays to nifti objects
        """

        convert = lambda vol : nib.Nifti1Image(vol, np.eye(4))
        fmri, mri = map(convert, (fmri, mri))

        return (fmri, mri)


    def extract_brain(self, fmri, mri):
        """
        Masks out the brain from the background
        """
        
        mask = nil.masking.compute_brain_mask(mri)
        mri  = mri * mask
        fmri = fmri * mask[..., None]

        return (fmri, mri)


    def smooth(self, fmri, mri):
        """
        Applies a gaussian filter
        """
        fmri, mri = nil.image.smooth_img(fmri), nil.image.smooth_img(mri)
        
        return (fmri, mri)
     