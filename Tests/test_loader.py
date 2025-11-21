"""
test_dataloader.py

Test suite for DataLoader class
Run from the project root with: python -m pytest Tests
"""

import pytest
import numpy as np
import nibabel as nib
import tempfile
import os
from unittest.mock import Mock, patch
from DataLoader import DataLoader


@pytest.fixture
def fmri_data():
    """Mock 4D fMRI (50x50x20x10 volumes)"""
    return np.random.randn(50, 50, 20, 10).astype(np.float32)


@pytest.fixture
def mri_data():
    """Mock 3D MRI (50x50x20)"""
    return np.random.randn(50, 50, 20).astype(np.float32)


@pytest.fixture
def fmri_nifti(fmri_data):
    return nib.Nifti1Image(fmri_data, np.eye(4))


@pytest.fixture
def mri_nifti(mri_data):
    return nib.Nifti1Image(mri_data, np.eye(4))


@pytest.fixture
def temp_dir(fmri_nifti, mri_nifti):
    """Directory with 2 fMRI/MRI pairs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fmri_dir = os.path.join(tmpdir, "fmri")
        mri_dir = os.path.join(tmpdir, "mri")
        os.makedirs(fmri_dir)
        os.makedirs(mri_dir)
        
        for i in range(2):
            nib.save(fmri_nifti, os.path.join(fmri_dir, f"sub_{i}.nii.gz"))
            nib.save(mri_nifti, os.path.join(mri_dir, f"sub_{i}.nii.gz"))
        
        yield tmpdir


class TestRegistration:
    """Test registration functions"""
    
    def test_register_vols_returns_array(self, mri_data):
        loader = DataLoader()
        v1 = mri_data
        v2 = mri_data + np.random.randn(*mri_data.shape) * 0.1
        
        with patch('DataLoader.HistogramRegistration') as mock_reg:
            mock_t = Mock()
            mock_t.as_affine.return_value = np.eye(4)
            mock_reg.return_value.optimize.return_value = mock_t
            
            result = loader.register_vols(v1, v2)
            assert isinstance(result, np.ndarray)
            assert result.shape == v2.shape
    
    def test_register_vols_with_transform(self, mri_data):
        loader = DataLoader()
        
        with patch('DataLoader.HistogramRegistration') as mock_reg:
            mock_t = Mock()
            mock_t.as_affine.return_value = np.eye(4)
            mock_reg.return_value.optimize.return_value = mock_t
            
            result, transform = loader.register_vols(mri_data, mri_data, return_transform=True)
            assert isinstance(result, np.ndarray)
            assert transform is not None
    
    def test_apply_affine_returns_array(self, mri_data):
        loader = DataLoader()
        mock_t = Mock()
        mock_t.as_affine.return_value = np.eye(4)
        
        result = loader.apply_affine(mri_data, mock_t)
        assert result is not None
        assert result.shape == mri_data.shape


class TestCoregistration:
    """Test volume coregistration"""
    
    def test_coregister_all_vols(self, fmri_nifti, mri_nifti):
        loader = DataLoader()
        
        with patch.object(loader, 'register_vols', return_value=np.random.randn(50, 50, 20)):
            fmri_result, mri_result = loader.coregister_all_vols(fmri_nifti, mri_nifti)
            
            assert isinstance(fmri_result, np.ndarray)
            assert isinstance(mri_result, np.ndarray)
            assert fmri_result.shape == fmri_nifti.shape
            assert mri_result.shape == mri_nifti.shape


class TestPreprocessing:
    """Test preprocessing steps"""
    
    @patch('DataLoader.nil.datasets.load_mni152_template')
    def test_normalize_to_template(self, mock_template, fmri_data, mri_data):
        loader = DataLoader()
        mock_template.return_value = Mock()
        
        with patch.object(loader, 'register_vols', return_value=(mri_data, np.eye(4))):
            fmri_result, mri_result = loader.normalize_to_template(fmri_data, mri_data)
            
            assert hasattr(loader, 'mri_t')
            assert hasattr(loader, 'fmri_t')
            mock_template.assert_called_once()
    
    def test_as_nifti(self, fmri_data, mri_data):
        loader = DataLoader()
        fmri_result, mri_result = loader.as_nifti(fmri_data, mri_data)
        
        assert isinstance(fmri_result, nib.Nifti1Image)
        assert isinstance(mri_result, nib.Nifti1Image)
        assert fmri_result.shape == fmri_data.shape
    
    @patch('DataLoader.nil.masking.compute_brain_mask')
    def test_extract_brain(self, mock_mask, fmri_nifti, mri_nifti):
        loader = DataLoader()
        mock_mask.return_value = nib.Nifti1Image(np.ones(mri_nifti.shape), np.eye(4))
        
        fmri_result, mri_result = loader.extract_brain(fmri_nifti, mri_nifti)
        
        mock_mask.assert_called_once()
        assert fmri_result.shape == fmri_nifti.shape
    
    @patch('DataLoader.nil.image.smooth_img')
    def test_smooth(self, mock_smooth, fmri_nifti, mri_nifti):
        loader = DataLoader()
        mock_smooth.side_effect = lambda x: x
        
        fmri_result, mri_result = loader.smooth(fmri_nifti, mri_nifti)
        
        assert mock_smooth.call_count == 2


class TestDataLoading:
    """Test data loading workflows"""
    
    def test_load_sample(self, fmri_nifti, mri_nifti):
        loader = DataLoader()
        loader.pipeline = []  # Disable pipeline for isolated test
        
        result = loader.load_sample(fmri_nifti, mri_nifti)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_callable_interface(self, fmri_nifti, mri_nifti):
        loader = DataLoader()
        loader.pipeline = []
        
        result = loader(fmri_nifti, mri_nifti)
        assert isinstance(result, tuple)
    
    def test_load_directory(self, temp_dir):
        loader = DataLoader()
        loader.pipeline = []
        
        samples = list(loader.load_directory(temp_dir))
        
        assert len(samples) == 2
        for sample in samples:
            assert isinstance(sample, tuple)
            assert len(sample) == 2


class TestIntegration:
    """End-to-end tests"""
    
    @patch('DataLoader.nil.datasets.load_mni152_template')
    @patch('DataLoader.nil.masking.compute_brain_mask')
    @patch('DataLoader.nil.image.smooth_img')
    def test_full_pipeline(self, mock_smooth, mock_mask, mock_template, 
                          fmri_nifti, mri_nifti):
        loader = DataLoader()
        
        mock_template.return_value = mri_nifti
        mock_mask.return_value = nib.Nifti1Image(np.ones(mri_nifti.shape), np.eye(4))
        mock_smooth.side_effect = lambda x: x
        
        with patch.object(loader, 'register_vols', return_value=(mri_nifti.get_fdata(), np.eye(4))):
            result = loader.load_sample(fmri_nifti, mri_nifti)
            
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2