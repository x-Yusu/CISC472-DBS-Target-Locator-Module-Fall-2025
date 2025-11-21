from DataLoader import DataLoader
import unittest

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from DataLoader import DataLoader 

class TestDataLoaderInit(unittest.TestCase):
    """Test cases for DataLoader initialization"""
    
    def test_init_creates_instance(self):
        """Test that DataLoader can be instantiated"""
        loader = DataLoader()
        self.assertIsInstance(loader, DataLoader)


class TestLoadSample(unittest.TestCase):
    """Test cases for load_sample method"""
    
    def test_load_sample_returns_numpy_array(self):
        """Test that load_sample returns a numpy array"""
        loader = DataLoader()
        result = loader.load_sample("dummy_path.nii")
        self.assertIsInstance(result, np.ndarray)
    
    def test_load_sample_shape(self):
        """Test that load_sample returns correct shape"""
        loader = DataLoader()
        result = loader.load_sample("dummy_path.nii")
        self.assertTrue( result.shape == (100, 100, 100, 20))
    
    def test_load_sample_with_none_path(self):
        """Test load_sample with None as path"""
        loader = DataLoader()
        result = loader.load_sample(None)
        self.assertIsInstance(result, np.ndarray)
    
    def test_load_sample_with_empty_string(self):
        """Test load_sample with empty string as path"""
        loader = DataLoader()
        result = loader.load_sample("")
        self.assertIsInstance(result, np.ndarray)


class TestLoadDirectory(unittest.TestCase):
    """Test cases for load_directory method"""
    
    def test_load_directory_is_generator(self):
        """Test that load_directory returns a generator"""
        loader = DataLoader()
        result = loader.load_directory("dummy_dir")
        assert hasattr(result, '__iter__') and hasattr(result, '__next__')
    
    def test_load_directory_yields_correct_count(self):
        """Test that load_directory yields correct number of items"""
        loader = DataLoader()
        results = list(loader.load_directory("dummy_dir"))
        self.assertEqual(len(results), 20)
    
    def test_load_directory_yields_numpy_arrays(self):
        """Test that all yielded items are numpy arrays"""
        loader = DataLoader()
        results = list(loader.load_directory("dummy_dir"))
        self.assertTrue(all(isinstance(item, np.ndarray) for item in results))
    
    def test_load_directory_yields_correct_shapes(self):
        """Test that all yielded arrays have correct shape"""
        loader = DataLoader()
        results = list(loader.load_directory("dummy_dir"))
        self.assertTrue(all(item.shape == (100, 100, 100, 20) for item in results))
    
    def test_load_directory_can_iterate_multiple_times(self):
        """Test that generator can be consumed"""
        loader = DataLoader()
        gen = loader.load_directory("dummy_dir")
        first_item = next(gen)
        second_item = next(gen)
        self.assertIsInstance(first_item, np.ndarray)
        self.assertIsInstance(second_item, np.ndarray)