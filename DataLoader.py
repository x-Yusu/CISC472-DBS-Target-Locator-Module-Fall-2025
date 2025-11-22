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

import SimpleITK as sitk

class DataLoader():

    def __init__(self):
        self.pipeline = [
            self.as_np,
            self.smooth,
            self.extract_brain,
            self.as_nifti,
            self.coregister_all_vols,
            self.normalize_to_template
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
        fmri_ref = self.register_vols(mri, np.mean(fmri_data,axis=3))
        
        # Register all fmris to the first fMRI
        for i in range(0, fmri.shape[-1]):
            if i < 10:
                fmri_data[...,i] = self.register_vols(fmri_ref, fmri_data[...,i])

        fmri = nib.Nifti1Image(fmri_data, np.eye(4))

        return (fmri, mri)


    def register_vols(self, v1, v2, mode='rigid', return_transform=False):
        v1, v2 = self.as_nifti(v1,v2)

        reg = HistogramRegistration(v1, v2)
        T = reg.optimize(mode)

        v2 = self.apply_affine(v2, T)

        if return_transform: return (v2, T)
        return v2


    def apply_affine(self, target, transform):
        if isinstance(target, nib.Nifti1Image):
            target = target.get_fdata()

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

        template = nil.datasets.load_mni152_template(resolution = 1.2)
        mri, mri_t = self.register_vols(template, mri, 'affine', True)

        fmri, mri = self.as_np(fmri, mri)

        # Find transform for one fMri
        fmri[...,0], fmri_t = self.register_vols(template, fmri[...,0], 'affine', True)

        # Register all fmris to the first fMRI
        for i in range(1, fmri.shape[-1]):
            fmri[...,i] = self.apply_affine(fmri[...,i], fmri_t)
            
        self.mri_t  = mri_t
        self.fmri_t = fmri_t

        return (fmri,mri)


    def as_nifti(self, fmri, mri):
        if isinstance(fmri, np.ndarray):
            fmri = nib.Nifti1Image(fmri, np.eye(4))

        if isinstance(mri, np.ndarray):
            mri = nib.Nifti1Image(mri, np.eye(4))

        return (fmri, mri)

    def as_np(self, fmri, mri):
        """ Casts the pair of fmri and mri nifti objects to numpy arrays """ 
        if isinstance(fmri, nib.Nifti1Image):
            fmri = fmri.get_fdata()

        if isinstance(mri, nib.Nifti1Image):
            mri = mri.get_fdata()

        return (fmri, mri)

    def extract_brain(self, fmri, mri):
        """ Masks out the brain from the background """
    
        def get_otsu_extration(vol):
            # Otsu threshold
            otsu = sitk.OtsuThresholdImageFilter()
            otsu.SetInsideValue(1)
            otsu.SetOutsideValue(0)
            mask = otsu.Execute(vol)
            
            # Clean up
            mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
            mask = sitk.BinaryFillhole(mask)
            
            return sitk.GetArrayFromImage(mask)

        fmri_array = sitk.GetArrayFromImage(fmri)
        mri_array  = sitk.GetArrayFromImage(mri)

        fmri_mean = np.mean(fmri_array, axis=3)
        fmri_ref = sitk.GetImageFromArray(fmri_mean)

        mask = get_otsu_extration(fmri_ref)
        fmri = fmri_array * mask[...,None]

        mask = get_otsu_extration(mri)
        mri  = mri_array * mask

        return (fmri, mri)


    def smooth(self, fmri, mri):
        """ Applies a gaussian filter """
        fmri = sitk.GetImageFromArray(fmri)
        mri  = sitk.GetImageFromArray(mri)

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(2.5)

        fmri = gaussian.Execute(fmri)
        mri = gaussian.Execute(mri)
        
        return (fmri, mri)
     