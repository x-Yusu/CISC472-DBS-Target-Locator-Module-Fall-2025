"""
test_dataloader.py

Minimal test suite using real fMRI data
Run: pytest test_dataloader.py -v
"""

import pytest
import numpy as np
import nibabel as nib
from nilearn import datasets
from DataLoader import DataLoader

import SimpleITK as sitk


@pytest.fixture(scope="module")
def fmri():
    """Load subject-matched fMRI from sample."""
    ds = datasets.fetch_development_fmri(n_subjects=1)
    img = nib.load(ds.func[0])
    data = img.get_fdata()[..., :10]
    return nib.Nifti1Image(data, img.affine)


@pytest.fixture(scope="module")
def mri():
    """Load the corresponding anatomical MRI for the same subject."""
    ds = datasets.fetch_development_fmri(n_subjects=1)

    img = nib.load(ds.func[0])
    data = img.get_fdata()[..., 0]
    data = np.roll(data, shift=5, axis=0)
    return nib.Nifti1Image(data, img.affine)


class TestRegistration:
    
    def test_register_vols(self, mri):
        loader = DataLoader()
        v1 = mri
        v2 = np.roll(v1.get_fdata(), shift=2, axis=0)
        v2 = nib.Nifti1Image(v2, np.eye(4))
        
        result = loader.register_vols(v1, v2)
        
        assert result.shape == v2.shape
        assert isinstance(result, np.ndarray)

    
    def test_apply_affine(self, mri):
        loader = DataLoader()
        v1 = mri.get_fdata()
        
        from unittest.mock import Mock
        mock_T = Mock()
        mock_T.as_affine.return_value = np.eye(4)
        
        result = loader.apply_affine(v1, mock_T)
        
        assert result.shape == v1.shape


class TestCoregistration:
    
    def test_coregister_all_vols(self, fmri, mri):
        loader = DataLoader()
        
        fmri_out, mri_out = loader.as_np(*loader.coregister_all_vols(fmri, mri))

        assert fmri_out.shape == fmri.shape
        assert mri_out.shape == mri.shape


class TestPreprocessing:
    
    def test_normalize_to_template(self, fmri, mri):
        loader = DataLoader()

        fmri_out, mri_out = loader.normalize_to_template(fmri, mri)
        
        assert hasattr(loader, 'mri_t')
        assert hasattr(loader, 'fmri_t')
        assert isinstance(fmri_out, np.ndarray)
    
    def test_as_nifti(self, fmri, mri):
        loader = DataLoader()
        
        fmri_nifti, mri_nifti = loader.as_nifti(fmri.get_fdata(), mri.get_fdata())
        
        assert isinstance(fmri_nifti, nib.Nifti1Image)
        assert isinstance(mri_nifti, nib.Nifti1Image)
    
    def test_extract_brain(self, fmri, mri):
        loader = DataLoader()
        
        fmri_masked, mri_masked = loader.extract_brain(sitk.GetImageFromArray(fmri.get_fdata()), sitk.GetImageFromArray(mri.get_fdata()))
        
        # Returns masked data
        assert fmri_masked is not None
        assert mri_masked is not None
    
    def test_smooth(self, fmri, mri):
        loader = DataLoader()
        
        fmri_smooth, mri_smooth = loader.smooth(fmri.get_fdata(), mri.get_fdata())
        
        # Returns smoothed images
        assert fmri_smooth is not None
        assert mri_smooth is not None


class TestPipeline:
    
    def test_load_sample(self, fmri, mri):
        loader = DataLoader()
        
        fmri_out, mri_out = loader.load_sample(fmri, mri)
        
        # Final output should be processed images
        assert fmri_out is not None
        assert mri_out is not None
    
    def test_callable(self, fmri, mri):
        loader = DataLoader()
        
        result = loader(fmri, mri)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_pipeline_exists(self):
        loader = DataLoader()
        
        assert hasattr(loader, 'pipeline')
        assert isinstance(loader.pipeline, list)
        assert len(loader.pipeline) > 0