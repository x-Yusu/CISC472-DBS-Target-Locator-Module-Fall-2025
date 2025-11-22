"""
masks.py
This file contains code for extracting appropriate masks and providing iterators to serve them

Author:
Leopold Ehrlich
"""

from nilearn import datasets, image
import nibabel as nib
from scipy.ndimage import binary_dilation

import numpy as np

def build_masks():
    """
    Extracts mean activity levels from brain structures using atlases.
    Uses anatomically proximal regions from available atlases as approximations for literature areas not available.

    Parameters:
        fmri - The fmri time series

    Returns:
        keys - Dict with keys as ROI names and values as scalars
    """

    # Load Harvard-Oxford subcortical atlas
    ho_atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    ho_img = nib.load(ho_atlas.filename)

    # Resample to 1.2 mm
    affine = np.diag([1.2, 1.2, 1.2, 1.0])
    target = image.new_img_like(ho_img, np.zeros((1,1,1)), affine=affine)
    ho_resampled = image.resample_to_img(ho_img, target, interpolation='nearest')

    ho_data = ho_img.get_fdata()

    print(ho_data)

    # Subgenual cingulate gyrus 
    # Approximate around anterior portion of Nucleus Accumbens
    acc_mask = (ho_data == 7) | (ho_data == 8)

    subgenual_mask = binary_dilation(acc_mask, iterations=3)
    # Filter to anterior regions only
    z_coords = np.where(subgenual_mask)

    anterior_threshold = np.percentile(z_coords[2], 60) 
    subgenual_mask = subgenual_mask & (np.arange(subgenual_mask.shape[2])[None, None, :] > anterior_threshold)

    roi['Subgenual cingulate'] = subgenual_mask

    # Ventral capsule/ventral striatum
    # Use area around putamen
    vs_mask = (ho_data == 11) | (ho_data == 12) 
    vs_mask = binary_dilation(vs_mask, iterations=2)
    roi_timeseries['Ventral striatum'] = vs_mask

    # Inferior thalamic peduncle
    # Thalamus and brainstem
    thal_mask = (ho_data == 16) | (ho_data == 17) | (ho_data == 15)

    # Take ventral portion
    thal_coords = np.where(thal_mask)

    ventral_threshold = np.percentile(thal_coords[1], 40)
    itp_mask = thal_mask & (np.arange(thal_mask.shape[1])[:, None, None] < ventral_threshold).T
    itp_mask = binary_dilation(itp_mask, iterations=1)

    roi_timeseries['Inferior thalamic peduncle'] = itp_mask

    # Medial forebrain bundle 
    # This is between the thalamus and nucleous accumben, so use intersection of dilation
    acc_mask = (ho_data == 7) | (ho_data == 8)
    thal_mask = (ho_data == 16) | (ho_data == 17)

    acc_dilated = binary_dilation(acc_mask, iterations=3)
    thal_dilated = binary_dilation(thal_mask, iterations=3)
    mfb_mask = acc_dilated & thal_dilated

    # Keep central/medial portions only
    x_center = mfb_mask.shape[0] // 2
    x_coords = np.arange(mfb_mask.shape[0])
    medial_band = (np.abs(x_coords - x_center) < mfb_mask.shape[0] * 0.15)
    mfb_mask = mfb_mask & medial_band[:, None, None]

    roi_timeseries['Medial forebrain bundle'] = mfb_mask

    return (roi_timeseries)

if __name__ == "__main__":
    dict_out = build_masks()

    with open('masks.json', 'wb') as mask_file:
        np.savez(dict_out, mask_file)