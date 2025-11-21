import os
import numpy as np
import nibabel as nib

def create_spherical_roi(shape, affine, mni_coords, radius_mm=6):
    """
    Function to create a spherical mask centered at MNI coordinates (mm)
    
    Args:
        shape (tuple): Image shape (x, y, z)
        affine (np.ndarray): Image Affine transformation matrix (4x4)
        mni_coords (tuple): Target MNI coordinates (x, y, z) [mm]
        radius_mm (float): Sphere radius [mm] (default is typically 6mm or 10mm)
    
    Returns:
        np.ndarray: Binary mask (0 or 1)
    """
    # 1. Calculate inverse Affine matrix (for MNI coords -> Voxel coords conversion)
    inv_affine = np.linalg.inv(affine)
    
    # 2. Extend MNI coords [x, y, z] to [x, y, z, 1] and transform
    mni_point = np.array([mni_coords[0], mni_coords[1], mni_coords[2], 1])
    voxel_center = inv_affine.dot(mni_point)[:3] # Transformed (i, j, k)
    
    # 3. Create grid
    # Create voxel coordinates for each voxel in the entire image
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Calculating distance by converting each voxel coord back to MNI space (mm) is costly, so calculate distance in voxel space simply (assuming isotropic voxels)
    # For accuracy, get voxel size and convert radius to voxel count
    
    # Get voxel size (scaling) (estimate from Affine matrix norm)
    vox_size_x = np.linalg.norm(affine[:3, 0])
    # vox_size_y = np.linalg.norm(affine[:3, 1])
    # vox_size_z = np.linalg.norm(affine[:3, 2])
    
    # Convert radius to voxel units
    radius_voxel = radius_mm / vox_size_x
    
    # Sphere equation: (x-cx)^2 + (y-cy)^2 + (z-cz)^2 <= r^2
    dist_sq = (x - voxel_center[0])**2 + (y - voxel_center[1])**2 + (z - voxel_center[2])**2
    
    mask = np.zeros(shape, dtype=np.uint8)
    mask[dist_sq <= radius_voxel**2] = 1
    
    return mask

def generate_masks_from_paper(reference_nii_path, output_dir):
    """
    Create and save masks based on coordinates in Table 2 of Gao et al. (2021)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load reference image (shape and Affine matrix required)
    img = nib.load(reference_nii_path)
    data_shape = img.shape
    affine = img.affine
    
    print(f"Reference Image Loaded: {reference_nii_path}")
    print(f"Shape: {data_shape}")

    # Gao et al. (2021) Table 2 coordinates
    # Source: fneur-12-751400.pdf 
    rois = {
        "ROI_Left_Mid_Cingulum": (-24, -27, 30),
        "ROI_Right_Precuneus":   (15, -48, 21),
        "ROI_Left_SFG":          (-21, 24, 24)
    }
    
    # Radius setting (typically 6mm, 8mm, 10mm are used in fMRI analysis)
    # Here we set it slightly wider at 6mm (12mm diameter)
    radius = 6.0 

    for name, coords in rois.items():
        print(f"Creating mask for {name} at MNI {coords} ...")
        
        mask_data = create_spherical_roi(data_shape, affine, coords, radius_mm=radius)
        
        # Voxel count check (confirm not empty)
        voxel_count = np.sum(mask_data)
        print(f" -> Generated mask with {voxel_count} voxels.")
        
        if voxel_count == 0:
            print("WARNING: Mask is empty! Check coordinates or affine matrix.")
        
        # Save as NIfTI
        mask_img = nib.Nifti1Image(mask_data, affine, img.header)
        save_path = os.path.join(output_dir, f"{name}.nii.gz")
        nib.save(mask_img, save_path)
        print(f" -> Saved to: {save_path}")

# Execution
if __name__ == "__main__":
    # Specify path of generated reference image
    ref_img_path = "../resources/ALFF_control_mean.nii.gz"
    
    # Mask save destination
    out_dir = "../resources/masks"
    
    if os.path.exists(ref_img_path):
        generate_masks_from_paper(ref_img_path, out_dir)
    else:
        print(f"Error: Reference image not found at {ref_img_path}")
        print("Please run generate_reference_maps.py first.")