# Visualizing comparisons between healthy and patient brains

This code compares fMRI data from patients with depression to that from healthy individuals to detect abnormal regions.

To start from analysis, either place patient data in data/test_patient_ALFF or specify a “relative path” as a command option.

```
Python test_volume_scoring.py -file [Relative Path]
Python test_whole_brain_analysis.py -file [Relative Path]
```

When running from the beginning, delete everything except the src folder and execute the py files in the order listed below.

1. Download the healthy control dataset (src/download_abide_controls.py). Download source: http://preprocessed-connectomes-project.org/abide/
2. Create an average fMRI map from the healthy control dataset (src/generate_reference_map.py).
3. Place the fMRI data of the depression patient in data/test_patient_ALFF.
4.1. Create a mask for the depression ROI (src/generate_mni_roi_masks.py). For ROI coordinates, refer to https://doi.org/10.3389/fneur.2021.751400.
4.2 Run src/test_volume_scoring.py to calculate the Z-score for the patient's ROI. 
$$Z = \frac{X - \mu}{\sigma}$$
5. Calculate the whole-brain Z-score, output a CSV file sorted in descending order by Z-score for each brain region, and further output a .nii.gz file mapping the Z-scores to 3D space (src/test_whole_brain_analysis.py).