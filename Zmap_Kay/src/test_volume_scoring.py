import os
import glob
import argparse
import numpy as np
import nibabel as nib

# 1. Score calculation logic
def compute_roi_scores(patient_volume, reference_mean_volume, reference_std_volume, roi_masks):
    # Calculate abnormality score (Z-score) and standard error metric for each ROI.
    
    results = {}
    
    for roi_name, mask in roi_masks.items():
        # Check mask and image size
        if patient_volume.shape != mask.shape:
            print(f"Warning: Shape mismatch for {roi_name}. Image: {patient_volume.shape}, Mask: {mask.shape}")
            continue

        # Extract voxel values within ROI
        pat_vals = patient_volume[mask > 0]
        ref_mean_vals = reference_mean_volume[mask > 0]
        ref_std_vals = reference_std_volume[mask > 0]
        
        if len(pat_vals) == 0:
            results[roi_name] = {"z_score": 0.0, "error_metric": 0.0, "voxel_count": 0}
            continue

        # Calculation
        patient_roi_mean = np.mean(pat_vals)
        baseline_roi_mean = np.mean(ref_mean_vals)
        baseline_roi_std = np.mean(ref_std_vals)
        
        # Z-score
        if baseline_roi_std < 1e-6:
            z_score = 0.0
        else:
            z_score = (patient_roi_mean - baseline_roi_mean) / baseline_roi_std

        # Standard Error (SEM)
        patient_roi_std = np.std(pat_vals)
        patient_roi_sem = patient_roi_std / np.sqrt(len(pat_vals))

        results[roi_name] = {
            "z_score": z_score,
            "error_metric": patient_roi_sem,
            "mean_activity": patient_roi_mean,
            "voxel_count": len(pat_vals)
        }
        
    return results

# 2. Load masks
def load_roi_masks(mask_dir):
    masks = {}
    mask_files = glob.glob(os.path.join(mask_dir, "*.nii.gz"))
    
    if not mask_files:
        print(f"Warning: No mask files found in {mask_dir}")
        return masks
        
    print(f"   Loading {len(mask_files)} masks from {mask_dir} ...")
    
    for fpath in mask_files:
        try:
            fname = os.path.basename(fpath)
            roi_name = fname.replace(".nii.gz", "").replace(".nii", "")
            img = nib.load(fpath)
            data = img.get_fdata()
            masks[roi_name] = data
        except Exception as e:
            print(f"   Error loading mask {fpath}: {e}")
            
    return masks

# 3. Main execution function
def run_integration_test(input_file_path=None):
    print("=== Starting ROI-based Analysis Integration Test ===\n")

    # Path settings
    # Based on the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resource directory
    ref_dir = os.path.join(script_dir, "../resources")
    mask_dir = os.path.join(script_dir, "../resources/masks")
    
    # Default patient data directory
    default_patient_dir = os.path.join(script_dir, "../data/test_patient_ALFF")
    
    ref_mean_path = os.path.join(ref_dir, "ALFF_control_mean.nii.gz")
    ref_std_path = os.path.join(ref_dir, "ALFF_control_std.nii.gz")
    
    # Determine patient file
    patient_path = ""
    
    if input_file_path:
        # If specified by argument
        if os.path.exists(input_file_path):
            patient_path = input_file_path
            print(f"Target File (from args): {patient_path}")
        else:
            print(f"Error: Specified file not found: {input_file_path}")
            return
    else:
        # If no argument, search in default folder
        print("No file specified. Searching in default directory...")
        patient_files = glob.glob(os.path.join(default_patient_dir, "*_alff.nii.gz"))
        if not patient_files:
            print(f"Error: No patient files found in {default_patient_dir}")
            return
        patient_path = patient_files[0] # Use the first one
        print(f"Target File (auto-detected): {patient_path}")

    # 1. Load Data
    print(f"\n1. Loading Volumes...")
    print(f"   Reference Mean: {os.path.basename(ref_mean_path)}")
    print(f"   Reference Std:  {os.path.basename(ref_std_path)}")

    if not (os.path.exists(ref_mean_path) and os.path.exists(ref_std_path)):
        print("Error: Reference maps missing. Run generate_reference_maps.py first.")
        return

    try:
        vol_mean = nib.load(ref_mean_path).get_fdata()
        vol_std = nib.load(ref_std_path).get_fdata()
        vol_pat = nib.load(patient_path).get_fdata()
        
        print(f"   Volumes loaded. Shape: {vol_pat.shape}")
        
        # Check image size consistency
        if vol_mean.shape != vol_pat.shape:
            print("Error: Reference and Patient volumes have different dimensions!")
            print("Note: This script assumes pre-registered data. Use the 'whole brain' script for auto-resampling.")
            return

    except Exception as e:
        print(f"Error loading volumes: {e}")
        return

    # 2. Load formal ROI masks
    print("\n2. Loading ROI Masks...")
    if not os.path.exists(mask_dir):
         print(f"Error: Mask directory not found: {mask_dir}")
         return

    roi_masks = load_roi_masks(mask_dir)
    
    if not roi_masks:
        print("Error: No masks loaded. Please run generate_mni_roi_masks.py first.")
        return

    # 3. Execute calculation
    print("\n3. Computing Scores...")
    scores = compute_roi_scores(vol_pat, vol_mean, vol_std, roi_masks)
    
    # 4. Display results
    print("\n" + "="*75)
    print(f"{'ROI Name':<30} | {'Z-Score':<10} | {'SEM':<10} | {'Voxels':<8}")
    print("-" * 75)
    
    # Sorting by score makes it easier to view (here sorted by name)
    for roi in sorted(scores.keys()):
        data = scores[roi]
        z = data['z_score']
        sem = data['error_metric']
        count = data['voxel_count']
        
        # Simple indicator mark based on Z-score
        mark = ""
        if abs(z) > 2.0: mark = "**" # Highlight abnormal value
        
        print(f"{roi:<30} | {z:>10.4f} {mark:<2} | {sem:>10.4f} | {count:>8}")
    
    print("=" * 75)
    print(f"Note: '**' indicates |Z-score| > 2.0 (potential abnormality)")

    # Simple verification
    if len(scores) > 0:
        print("\nSUCCESS: Calculation completed for all loaded masks.")
    else:
        print("\nFAILED: No scores computed.")

if __name__ == "__main__":
    # Command line argument settings
    parser = argparse.ArgumentParser(description="Calculate ROI-based Z-scores from fMRI ALFF data.")
    
    # Allow specifying path with -file or --file
    parser.add_argument("-file", "--file", type=str, help="Path to the patient .nii.gz file to analyze", required=False)
    
    args = parser.parse_args()
    
    # Execute analysis
    run_integration_test(args.file)