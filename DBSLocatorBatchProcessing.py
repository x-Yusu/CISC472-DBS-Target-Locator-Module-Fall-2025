import os
import numpy as np
import pandas as pd
import h5py
import logging

# Configure logging to show info in the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_tian_label_mapping():
    """
    Returns a dictionary mapping Tian Scale II Label IDs (1-32) to Region Names.
    """
    roi_names = {
        # --- RIGHT HEMISPHERE (1-16) ---
        1: "Anterior Hippocampus (Right)",
        2: "Posterior Hippocampus (Right)",
        3: "Lateral Amygdala (Right)",
        4: "Medial Amygdala (Right)",
        5: "Dorsoposterior Thalamus (Right)",
        6: "Ventroposterior Thalamus (Right)",
        7: "Ventroanterior Thalamus (Right)",
        8: "Dorsoanterior Thalamus (Right)",
        9: "Nucleus Accumbens Shell (Right)",
        10: "Nucleus Accumbens Core (Right)",
        11: "Posterior Globus Pallidus (Right)",
        12: "Anterior Globus Pallidus (Right)",
        13: "Anterior Putamen (Right)",
        14: "Posterior Putamen (Right)",
        15: "Anterior Caudate (Right)",
        16: "Posterior Caudate (Right)",

        # --- LEFT HEMISPHERE (17-32) ---
        17: "Anterior Hippocampus (Left)",
        18: "Posterior Hippocampus (Left)",
        19: "Lateral Amygdala (Left)",
        20: "Medial Amygdala (Left)",
        21: "Dorsoposterior Thalamus (Left)",
        22: "Ventroposterior Thalamus (Left)",
        23: "Ventroanterior Thalamus (Left)",
        24: "Dorsoanterior Thalamus (Left)",
        25: "Nucleus Accumbens Shell (Left)",
        26: "Nucleus Accumbens Core (Left)",
        27: "Posterior Globus Pallidus (Left)",
        28: "Anterior Globus Pallidus (Left)",
        29: "Anterior Putamen (Left)",
        30: "Posterior Putamen (Left)",
        31: "Anterior Caudate (Left)",
        32: "Posterior Caudate (Left)"
    }
    return roi_names


def load_h5_data(path):
    """
    Helper to load data from .h5 file regardless of key name.
    """
    try:
        df = pd.read_hdf(path)
        return df.values
    except:
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            return f[key][:]


def get_raw_fc_matrix(path):
    """
    Returns the raw functional connectivity correlation matrix for a given file.
    """
    data = load_h5_data(path)
    # Transpose if necessary (expecting 434 regions)
    if data.shape[0] == 434:
        data = data.T

    # Calculate Correlation Matrix
    fc = pd.DataFrame(data).corr().values
    return fc


def compute_subject_metric(subject_folder):
    """
    Computes the connectivity metric for a single subject folder.
    Logic:
    1. Find all 'rest' .h5 files (AP/PA).
    2. Load raw correlation matrices for each.
    3. Average the matrices (cancels noise).
    4. Compute mean absolute connectivity strength from averaged matrix.
    """
    if not os.path.exists(subject_folder):
        logging.warning(f"Subject folder not found: {subject_folder}")
        return None

    h5_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder)
                if f.endswith('.h5') and 'rest' in f]

    if not h5_files:
        logging.warning(f"No resting state .h5 files in {subject_folder}")
        return None

    matrices = []
    for f in h5_files:
        try:
            fc = get_raw_fc_matrix(f)
            matrices.append(fc)
        except Exception as e:
            logging.error(f"Error loading {f}: {e}")

    if not matrices:
        return None

    # Average the raw correlation matrices first (Triangle Inequality Fix)
    avg_matrix = np.mean(np.array(matrices), axis=0)

    # Calculate Strength: Mean of absolute correlations for each region
    # Only keep the 434 regions
    metric = np.mean(np.abs(avg_matrix), axis=1)

    return metric


def process_batch(control_root_dir, patient_root_dir, output_csv):
    """
    Main processing function.
    1. Computes baseline stats from Healthy Controls.
    2. Computes Z-scores for every Patient.
    3. Exports results to CSV.
    """

    # --- 1. Compute Control Baseline ---
    logging.info("--- Processing Healthy Controls ---")
    control_metrics = []

    # Walk through control directory
    # Assumes structure: data/healthy/SubjectID/file.h5
    control_subjects = [os.path.join(control_root_dir, d) for d in os.listdir(control_root_dir)
                        if os.path.isdir(os.path.join(control_root_dir, d))]

    if not control_subjects:
        logging.error(f"No subject directories found in {control_root_dir}")
        return

    for subject_path in control_subjects:
        metric = compute_subject_metric(subject_path)
        if metric is not None:
            control_metrics.append(metric)
            # logging.info(f"Processed Control: {os.path.basename(subject_path)}")

    if not control_metrics:
        logging.error("Failed to compute any control metrics.")
        return

    control_metrics = np.array(control_metrics)
    ctrl_mean = np.mean(control_metrics, axis=0)
    ctrl_std = np.std(control_metrics, axis=0)

    # Avoid division by zero
    ctrl_std[ctrl_std == 0] = 1e-9

    logging.info(f"Baseline calculated from {len(control_metrics)} healthy controls.")

    # --- 2. Process Patients ---
    logging.info("--- Processing Patients ---")
    patient_subjects = [os.path.join(patient_root_dir, d) for d in os.listdir(patient_root_dir)
                        if os.path.isdir(os.path.join(patient_root_dir, d))]

    if not patient_subjects:
        logging.error(f"No subject directories found in {patient_root_dir}")
        return

    # Prepare DataFrame for results
    roi_map = get_tian_label_mapping()
    results_list = []

    for subject_path in patient_subjects:
        subject_id = os.path.basename(subject_path)
        patient_metric = compute_subject_metric(subject_path)

        if patient_metric is not None:
            # Calculate Z-Scores
            z_scores = (patient_metric - ctrl_mean) / ctrl_std
            z_scores = np.nan_to_num(z_scores)

            # Extract Subcortical Regions (Indices 400-431 for Tian Scale 2)
            # Indices are 0-based, so region 1 is index 400
            subcortex_z = z_scores[400:432]

            # Create a dictionary for this patient row
            row = {'Patient_ID': subject_id}

            for i in range(32):
                label_id = i + 1
                region_name = roi_map.get(label_id, f"Region_{label_id}")
                row[region_name] = subcortex_z[i]

            results_list.append(row)
            # logging.info(f"Processed Patient: {subject_id}")

    # --- 3. Export to CSV ---
    if results_list:
        df_results = pd.DataFrame(results_list)

        # Reorder columns to put Patient_ID first, then regions sorted by name or ID
        cols = ['Patient_ID'] + [col for col in df_results.columns if col != 'Patient_ID']
        df_results = df_results[cols]

        df_results.to_csv(output_csv, index=False)
        logging.info(f"--- SUCCESS ---")
        logging.info(f"Processed {len(results_list)} patients.")
        logging.info(f"Results saved to: {output_csv}")
    else:
        logging.warning("No patient results to save.")


if __name__ == "__main__":
    # --- CONFIGURATION ---

    CONTROL_DATA_PATH = r"data/healthy"
    PATIENT_DATA_PATH = r"data/patient"
    OUTPUT_FILE = r"DBS_ZScore_Analysis_Results.csv"

    # Validate paths before running
    if not os.path.exists(CONTROL_DATA_PATH):
        print(f"Error: Control path not found: {CONTROL_DATA_PATH}")
        # Create dummy folders for testing if they don't exist? (Optional)
    elif not os.path.exists(PATIENT_DATA_PATH):
        print(f"Error: Patient path not found: {PATIENT_DATA_PATH}")
    else:
        process_batch(CONTROL_DATA_PATH, PATIENT_DATA_PATH, OUTPUT_FILE)