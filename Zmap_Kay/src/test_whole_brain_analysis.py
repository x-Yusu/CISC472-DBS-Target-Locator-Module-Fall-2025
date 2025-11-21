import os
import glob
import argparse  # For argument processing
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import coord_transform
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Calculation logic (with peak detection and coordinate transformation)
def compute_atlas_scores_with_peaks(patient_img, ref_mean_img, ref_std_img, atlas_img, labels):
    # Calculate scores (Mean Z, Peak Z, Peak Coordinates) for each whole brain region.
    
    vol_pat = patient_img.get_fdata()
    vol_mean = ref_mean_img.get_fdata()
    vol_std = ref_std_img.get_fdata()
    vol_atlas = atlas_img.get_fdata()
    affine = patient_img.affine
    
    results = []
    unique_ids = np.unique(vol_atlas)
    unique_ids = unique_ids[unique_ids > 0]
    
    print(f"Analyzing {len(unique_ids)} regions...")

    for region_id in unique_ids:
        region_id = int(region_id)
        
        # Get label name
        try: 
            region_name = labels[region_id]
        except: 
            region_name = f"Region_{region_id}"

        # Create mask
        mask = (vol_atlas == region_id)
        pat_vals = vol_pat[mask]
        mean_vals = vol_mean[mask]
        std_vals = vol_std[mask]
        
        if len(pat_vals) == 0: continue

        base_std_mean = np.mean(std_vals)
        if base_std_mean < 1e-6: continue
            
        # Mean Z-score
        mean_z = (np.mean(pat_vals) - np.mean(mean_vals)) / base_std_mean
        
        # Search for peak (maximum abnormal value)
        voxel_z_scores = (pat_vals - mean_vals) / (std_vals + 1e-9)
        max_idx_local = np.argmax(np.abs(voxel_z_scores))
        peak_z = voxel_z_scores[max_idx_local]
        
        # Coordinate transformation (Index -> MNI)
        mask_coords = np.where(mask)
        peak_x = mask_coords[0][max_idx_local]
        peak_y = mask_coords[1][max_idx_local]
        peak_z_idx = mask_coords[2][max_idx_local]
        
        mni_x, mni_y, mni_z = coord_transform(peak_x, peak_y, peak_z_idx, affine)
        
        results.append({
            "ID": region_id,          # Required for Z-map creation
            "Region": region_name,
            "Mean_Z": mean_z,         # Region mean abnormality
            "Peak_Z": peak_z,         # Max abnormality within region
            "Abs_Mean_Z": abs(mean_z),# For ranking
            "Peak_MNI": f"({mni_x:.1f}, {mni_y:.1f}, {mni_z:.1f})",
            "Voxels": len(pat_vals)
        })
        
    return pd.DataFrame(results)

# 2. Z-map save function
def save_z_score_map(df_results, atlas_img, output_filename):
    """
    Fill each atlas region with calculated Z-score and save as 3D NIfTI.
    """
    print(f"Generating Z-Score Map: {output_filename} ...")
    atlas_data = atlas_img.get_fdata()
    
    # Set background to NaN (to make background transparent when displaying in Slicer)
    z_map_data = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    
    for index, row in df_results.iterrows():
        region_id = row['ID']
        z_score = row['Mean_Z']
        # Fill location of corresponding region ID with Z-score
        z_map_data[atlas_data == region_id] = z_score
        
    z_map_img = nib.Nifti1Image(z_map_data, atlas_img.affine, atlas_img.header)
    nib.save(z_map_img, output_filename)
    print(" -> Z-Map Saved.")

# 3. Main execution block
def run_analysis(input_file_path=None):
    print("=== Starting Full Brain Analysis (CSV & Z-Map) ===\n")

    # Reference data path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(script_dir, "../resources")
    
    # Default patient data folder (used if no argument)
    default_patient_dir = os.path.join(script_dir, "../data/test_patient_ALFF")
    
    # Output directory
    output_dir = os.path.join(script_dir, "../output")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Reference file paths
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
        patient_path = patient_files[0]
        print(f"Target File (auto-detected): {patient_path}")

    # 1. Loading Data & Resampling
    print("\n1. Loading Data & Resampling...")
    if not (os.path.exists(ref_mean_path) and os.path.exists(ref_std_path)):
        print(f"Error: Reference maps not found in {ref_dir}")
        return

    try:
        # Download AAL atlas
        aal = datasets.fetch_atlas_aal(version='SPM12')
        
        # Load patient image
        img_pat = nib.load(patient_path)

        # Resample reference image (match to patient image)
        # Force resize with force_resample=True
        img_ref_mean = image.resample_to_img(
            nib.load(ref_mean_path), 
            img_pat, 
            force_resample=True
        )
        img_ref_std = image.resample_to_img(
            nib.load(ref_std_path), 
            img_pat, 
            force_resample=True
        )
        
        # Resample atlas (use Nearest Neighbor method)
        img_atlas_res = image.resample_to_img(
            aal.maps, 
            img_pat, 
            interpolation='nearest', 
            force_resample=True
        )
    except Exception as e:
        print(f"Error during loading/resampling: {e}")
        return

    # Execute calculation
    print("\n2. Computing Scores...")
    df_results = compute_atlas_scores_with_peaks(
        img_pat, img_ref_mean, img_ref_std, img_atlas_res, aal.labels
    )

    # Output result 1: Console display
    print("\n" + "="*80)
    print("TOP 5 ABNORMAL REGIONS (Sorted by |Mean Z|)")
    print("="*80)
    # Display top 5
    df_sorted = df_results.sort_values(by="Abs_Mean_Z", ascending=False)
    print(df_sorted[["Region", "Mean_Z", "Peak_MNI"]].head(5).to_string(index=False))
    print("-" * 80)

    # Output result 2: Save CSV
    # Generate output filename based on input filename
    base_name = os.path.splitext(os.path.splitext(os.path.basename(patient_path))[0])[0]
    csv_filename = os.path.join(output_dir, f"{base_name}_analysis.csv")
    df_sorted.to_csv(csv_filename, index=False)
    print(f"\n[CSV Output] Saved results to: {csv_filename}")

    # Output result 3: Save Z-map
    zmap_filename = os.path.join(output_dir, f"{base_name}_z_map.nii.gz")
    save_z_score_map(df_results, img_atlas_res, zmap_filename)
    print(f"[Z-Map Output] Saved 3D map to: {zmap_filename}")

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    # Command line argument settings
    parser = argparse.ArgumentParser(description="Calculate Z-scores for brain regions from fMRI ALFF data.")
    
    # Allow specifying path with -file or --file
    parser.add_argument("-file", "--file", type=str, help="Path to the patient .nii.gz file to analyze", required=False)
    
    args = parser.parse_args()
    
    # Execute analysis
    run_analysis(args.file)