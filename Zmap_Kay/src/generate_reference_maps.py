import os
import glob
import numpy as np
import nibabel as nib

def generate_reference_maps(input_dir, output_dir, mask_path=None):
    """
    Function to load all ALFF maps (NIfTI) in the specified folder and create and save mean and standard deviation maps.

    Args:
        input_dir (str): Folder path containing healthy control ALFF NIfTI files (.nii or .nii.gz)
        output_dir (str): Folder path to save generated mean and standard deviation maps
        mask_path (str, optional): Path to binary mask to restrict analysis region (optional)
    """
    
    # 1. Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Get list of target files (.nii or .nii.gz)
    #    Use glob.glob(..., recursive=True) to search recursively
    search_pattern = os.path.join(input_dir, "*.nii*")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    
    print(f"Found {len(file_list)} files. Starting processing...")

    # 3. Load and stack data
    #    For memory efficiency, first collect in a list then convert to numpy array
    data_list = []
    first_img = None # For saving header info (Affine matrix etc.)
    
    for i, fpath in enumerate(file_list):
        try:
            img = nib.load(fpath)
            data = img.get_fdata()
            
            # Keep first image info as reference
            if first_img is None:
                first_img = img
                ref_shape = data.shape
                print(f"Reference Reference shape: {ref_shape}")
            
            # Shape check (all images must have same resolution)
            if data.shape != ref_shape:
                print(f"Warning: Skipping {os.path.basename(fpath)} due to shape mismatch {data.shape}")
                continue
                
            data_list.append(data)
            
            # Display progress
            if (i + 1) % 10 == 0:
                print(f"Loaded {i + 1}/{len(file_list)} files...")
                
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    if not data_list:
        raise RuntimeError("No valid data loaded.")

    # Convert list to 4D array, stack as (X, Y, Z, Subject)
    # It is common to add subject dimension to axis=-1 (last dimension)
    stacked_data = np.stack(data_list, axis=-1)
    print(f"Data stacking complete. Shape: {stacked_data.shape}")

    # 4. Calculate Mean and Std
    print("Calculating Mean and Std...")
    
    # Calculate along the last dimension (subject direction)
    mean_data = np.mean(stacked_data, axis=-1)
    std_data = np.std(stacked_data, axis=-1)

    # Set values outside mask to 0 if mask exists
    # if mask_path and os.path.exists(mask_path):
    #     mask_img = nib.load(mask_path)
    #     mask_data = mask_img.get_fdata()
    #     # Perform mask resizing or resampling here if necessary
    #     if mask_data.shape == mean_data.shape:
    #         mean_data[mask_data == 0] = 0
    #         std_data[mask_data == 0] = 0
    #         print("Applied masking.")
    #     else:
    #         print("Warning: Mask shape mismatch. Skipping masking.")

    # 5. Save as NIfTI images
    #    Use header and Affine matrix from the first image
    mean_img = nib.Nifti1Image(mean_data, first_img.affine, first_img.header)
    std_img = nib.Nifti1Image(std_data, first_img.affine, first_img.header)

    mean_output_path = os.path.join(output_dir, "ALFF_control_mean.nii.gz")
    std_output_path = os.path.join(output_dir, "ALFF_control_std.nii.gz")

    nib.save(mean_img, mean_output_path)
    nib.save(std_img, std_output_path)

    print(f"Successfully saved reference maps:")
    print(f" - Mean: {mean_output_path}")
    print(f" - Std:  {std_output_path}")

# Execution
if __name__ == "__main__":
    input_folder_path = "../data/healthy_controls_ALFF"
    output_folder_path = "../resources"
    
    # Execution
    generate_reference_maps(input_folder_path, output_folder_path)