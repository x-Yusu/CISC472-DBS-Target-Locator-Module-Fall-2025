import os
import pandas as pd
import urllib.request
import urllib.error

def download_abide_controls_alff(output_dir, max_subjects=None):
    """
    Script to download ALFF data for healthy controls (Control) from ABIDE I
    
    Args:
        output_dir (str): Destination folder
        max_subjects (int): Maximum number of subjects to download (set integer value. Set to None for all)
    """
    
    # 1. Create destination directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Get Phenotypic information (participant list)
    pheno_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
    print("Fetching phenotypic data...")
    
    try:
        # Read directly with pandas
        df = pd.read_csv(pheno_url)
    except Exception as e:
        print(f"Error fetching phenotype CSV: {e}")
        return

    # 3. Filter only healthy controls (DX_GROUP == 2)
    # DX_GROUP: 1=Autism, 2=Control
    controls_df = df[df['DX_GROUP'] == 2]
    print(f"Found {len(controls_df)} control subjects in total.")
    
    # Limit number of subjects if necessary
    if max_subjects is not None:
        controls_df = controls_df.head(max_subjects)
        print(f"Downloading first {max_subjects} subjects...")

    # 4. Execute download
    # Settings: Pipeline=cpac, Strategy=filt_global (Filtering + Global Signal Regression)
    base_url_template = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/alff/{}_alff.nii.gz"
    
    success_count = 0
    
    for index, row in controls_df.iterrows():
        file_id = row['FILE_ID']
        
        # Skip if FILE_ID is no_filename
        if file_id == "no_filename":
            continue

        download_url = base_url_template.format(file_id)
        output_filename = os.path.join(output_dir, f"{file_id}_alff.nii.gz")

        if os.path.exists(output_filename):
            print(f"[Skipping] Already exists: {output_filename}")
            success_count += 1
            continue

        print(f"Downloading {file_id} ...")
        
        try:
            urllib.request.urlretrieve(download_url, output_filename)
            success_count += 1
        except urllib.error.HTTPError as e:
            print(f"  Failed to download {file_id}: {e.code} (File might not exist for this pipeline)")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDownload complete. Successfully downloaded {success_count} files to {output_dir}")

# Execution
if __name__ == "__main__":
    # Specify save destination
    save_dir = "../data/healthy_controls_ALFF"
    
    # Execute (only specified number of subjects. Specify max_subjects=None if you want all)
    download_abide_controls_alff(save_dir, max_subjects=20)