import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# --- CONFIGURATION ---
INPUT_CSV = "DBS_ZScore_Analysis_Results.csv"  # The file created by BatchProcessing.py
OUTPUT_PDF = "DBS_Patient_Distributions_Report.pdf"


def generate_report(csv_path, pdf_path):
    """
    Reads the patient Z-score CSV and generates a multi-page PDF report
    containing distribution plots for each brain region.
    """

    if not os.path.exists(csv_path):
        print(f"Error: Input file not found: {csv_path}")
        print("Please run BatchProcessing.py first to generate the results.")
        return

    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Identify region columns (exclude 'Patient_ID')
    region_cols = [col for col in df.columns if col != 'Patient_ID']

    if not region_cols:
        print("Error: No region data found in CSV.")
        return

    print(f"Found {len(df)} patients and {len(region_cols)} regions.")
    print("-" * 40)
    print("Generating statistical summary...")
    print(df[region_cols].describe().transpose())  # Print stats to console
    print("-" * 40)

    print(f"Creating distribution plots in {pdf_path}...")

    # Use a style for the plots
    sns.set_theme(style="whitegrid")

    with PdfPages(pdf_path) as pdf:
        # Create a plot for each region
        for region in region_cols:
            plt.figure(figsize=(10, 6))

            # Histogram with KDE
            # We use a fixed range if possible, or let it auto-scale
            sns.histplot(df[region], kde=True, bins=15, color='skyblue', edgecolor='black')

            # Add reference lines for standard Z-score thresholds
            plt.axvline(x=-1.96, color='r', linestyle='--', label='Significance (p=0.05)')
            plt.axvline(x=1.96, color='r', linestyle='--')
            plt.axvline(x=0, color='k', linestyle='-', linewidth=1, label='Mean')

            plt.title(f"Z-Score Distribution: {region}", fontsize=14)
            plt.xlabel("Z-Score (Deviation from Healthy Control Baseline)", fontsize=12)
            plt.ylabel("Count (Number of Patients)", fontsize=12)
            plt.legend()

            # Adjust layout and save to PDF page
            plt.tight_layout()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()  # close the figure to free memory

    print(f"Success! Report saved to: {os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    generate_report(INPUT_CSV, OUTPUT_PDF)