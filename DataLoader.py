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

class DataLoader():

    def __init__(self):
        self.pipeline = [
            self.smooth,
            self.coregister_vols,
            self.extract_brain
            ]


    def __call__(self, *args):
        return self.load_sample(*args)

    
    def load_sample(self, fmri, mri):
        """
        Loads an FMRI file, runs preprocessing and returns it as a numpy array
        """

        if isinstance(fmri, np.ndarray):
            as_nifti = lambda x : nib.Nifti1Image(x, np.eye(4))
            fmri, mri = map(as_nifti, (fmri, mri))

        elif not isinstance(fmri, nibabel.nifti1.Nifti1Image):
            raise ValueError("Inputs must be of type np.ndarray or nibabel.nifti1.Nifti1Image")

        for op in self.pipeline:
            fmri, mri = op(fmri, mri)

        return (fmri, mri)


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

    def coregister_vols(self, fmri, mri):
        fmri_data = fmri.get_fdata()
        n_volumes = fmri_data.shape[3]
        
        # Resample each volume
        output_data = []
        for t in range(n_volumes):
            vol_img = nib.Nifti1Image(
                fmri_data[:, :, :, t],
                fmri.affine,
                fmri.header
            )
            
            vol_resampled = resample_to_img(
                vol_img,
                mri,
                interpolation='continuous'
            )
            
            output_data.append(vol_resampled.get_fdata())
        
        # Stack into 4D
        output_4d = np.stack(output_data, axis=-1)
        func_coreg = nib.Nifti1Image(
            output_4d,
            self.anat_brain.affine,
            self.anat_brain.header
        )