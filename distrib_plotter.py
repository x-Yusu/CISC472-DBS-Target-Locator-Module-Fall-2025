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
    stats_df = df[region_cols].describe().transpose()  # Calculate stats once
    print(stats_df)  # Print stats to console
    print("-" * 40)

    print(f"Creating distribution plots in {pdf_path}...")

    # Use a style for the plots
    sns.set_theme(style="whitegrid")

    with PdfPages(pdf_path) as pdf:
        # Create a plot for each region
        for region in region_cols:
            # Create figure with extra width for stats text
            plt.figure(figsize=(12, 6))

            # Create a grid for layout: Left=Plot, Right=Text
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])

            # --- PLOT AREA (Left) ---
            ax_plot = plt.subplot(gs[0])

            # Histogram with KDE
            sns.histplot(df[region], kde=True, bins=15, color='skyblue', edgecolor='black', ax=ax_plot)

            # Add reference lines for standard Z-score thresholds
            ax_plot.axvline(x=-1.96, color='r', linestyle='--', label='Significance (p=0.05)')
            ax_plot.axvline(x=1.96, color='r', linestyle='--')
            ax_plot.axvline(x=0, color='k', linestyle='-', linewidth=1, label='Mean')

            ax_plot.set_title(f"Z-Score Distribution: {region}", fontsize=14)
            ax_plot.set_xlabel("Z-Score (Deviation from Healthy Control Baseline)", fontsize=12)
            ax_plot.set_ylabel("Count (Number of Patients)", fontsize=12)
            ax_plot.legend()

            # --- STATS TEXT AREA (Right) ---
            ax_text = plt.subplot(gs[1])
            ax_text.axis('off')  # Hide axes for text area

            # Get stats for this region
            r_stats = stats_df.loc[region]

            stats_text = (
                f"Statistical Summary\n"
                f"-------------------\n\n"
                f"Count: {int(r_stats['count'])}\n\n"
                f"Mean:  {r_stats['mean']:.4f}\n"
                f"Std:   {r_stats['std']:.4f}\n\n"
                f"Min:   {r_stats['min']:.4f}\n"
                f"25%:   {r_stats['25%']:.4f}\n"
                f"50%:   {r_stats['50%']:.4f}\n"
                f"75%:   {r_stats['75%']:.4f}\n"
                f"Max:   {r_stats['max']:.4f}"
            )

            # Place text in the center of the right panel
            ax_text.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                         verticalalignment='center', transform=ax_text.transAxes,
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

            # Adjust layout and save to PDF page
            plt.tight_layout()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()  # close the figure to free memory

    print(f"Success! Report saved to: {os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    generate_report(INPUT_CSV, OUTPUT_PDF)