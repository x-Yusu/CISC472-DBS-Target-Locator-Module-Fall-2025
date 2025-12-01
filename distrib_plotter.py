import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_CSV = "DBS_ZScore_Analysis_Results.csv"  # The file created by BatchProcessing.py
OUTPUT_PDF = "DBS_Patient_Distributions_Report.pdf"


def generate_report(csv_path, pdf_path):
    """
    Reads the patient Z-score CSV and generates a multi-page PDF report.
    Includes:
    1. Overall Distribution (All Regions)
    2. Rank 1 Region Distribution (Pie Chart) - NEW
    3. Individual Region Distributions (Histograms)
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

    # Calculate stats for individual regions
    stats_df = df[region_cols].describe().transpose()

    # --- AGGREGATE DATA FOR OVERALL DISTRIBUTION ---
    all_z_scores = df[region_cols].values.flatten()
    all_z_scores = all_z_scores[~np.isnan(all_z_scores)]

    all_z_series = pd.Series(all_z_scores, name="All Regions Combined")
    overall_stats = all_z_series.describe()

    # --- RANK 1 ANALYSIS (For Pie Chart) ---
    # Find the column name with the max absolute Z-score for each row
    # We use abs() because we care about the most abnormal region, whether hypo or hyper
    rank1_regions = df[region_cols].abs().idxmax(axis=1)

    # Count frequency of each region being Rank 1
    rank1_counts = rank1_regions.value_counts()

    print("-" * 40)
    print("Top Rank 1 Regions (Most Abnormal):")
    print(rank1_counts.head())
    print("-" * 40)

    print(f"Creating distribution plots in {pdf_path}...")

    # Use a style for the plots
    sns.set_theme(style="whitegrid")

    with PdfPages(pdf_path) as pdf:

        # --- PAGE 1: PIE CHART (Rank 1 Regions) ---
        plt.figure(figsize=(10, 8))

        # Plot Pie Chart
        # autopct shows percentage
        # pctdistance moves the percentage text
        wedges, texts, autotexts = plt.pie(
            rank1_counts,
            labels=rank1_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.85,
            textprops={'fontsize': 9}
        )

        # Draw circle for Donut Chart style (optional, looks cleaner)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        plt.title(f"Most Abnormal Brain Regions (Rank 1)\nAcross {len(df)} Patients", fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- PAGE 2: OVERALL Z-SCORE DISTRIBUTION ---
        plt.figure(figsize=(12, 6))
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])

        # Plot
        ax_plot = plt.subplot(gs[0])
        sns.histplot(all_z_scores, kde=True, bins=30, color='purple', edgecolor='black', ax=ax_plot)

        ax_plot.axvline(x=-1.96, color='r', linestyle='--', label='Significance (p=0.05)')
        ax_plot.axvline(x=1.96, color='r', linestyle='--')
        ax_plot.axvline(x=0, color='k', linestyle='-', linewidth=1, label='Mean')

        ax_plot.set_title("Overall Brain Activity Deviation (All Regions Combined)", fontsize=14, fontweight='bold')
        ax_plot.set_xlabel("Z-Score (Deviation from Healthy Control Baseline)", fontsize=12)
        ax_plot.set_ylabel("Count (Total Region Samples)", fontsize=12)
        ax_plot.legend()

        # Stats Text
        ax_text = plt.subplot(gs[1])
        ax_text.axis('off')

        stats_text = (
            f"Overall Summary\n"
            f"(All Patients & Regions)\n"
            f"----------------------\n\n"
            f"Count: {int(overall_stats['count'])}\n\n"
            f"Mean:  {overall_stats['mean']:.4f}\n"
            f"Std:   {overall_stats['std']:.4f}\n\n"
            f"Min:   {overall_stats['min']:.4f}\n"
            f"25%:   {overall_stats['25%']:.4f}\n"
            f"50%:   {overall_stats['50%']:.4f}\n"
            f"75%:   {overall_stats['75%']:.4f}\n"
            f"Max:   {overall_stats['max']:.4f}"
        )

        ax_text.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                     verticalalignment='center', transform=ax_text.transAxes,
                     bbox=dict(boxstyle="round,pad=0.5", fc="lavender", ec="purple", alpha=0.9))

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- PAGE 3+: INDIVIDUAL REGION PAGES ---
        for region in region_cols:
            plt.figure(figsize=(12, 6))
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])

            # Plot
            ax_plot = plt.subplot(gs[0])
            sns.histplot(df[region], kde=True, bins=15, color='skyblue', edgecolor='black', ax=ax_plot)

            ax_plot.axvline(x=-1.96, color='r', linestyle='--', label='Significance (p=0.05)')
            ax_plot.axvline(x=1.96, color='r', linestyle='--')
            ax_plot.axvline(x=0, color='k', linestyle='-', linewidth=1, label='Mean')

            ax_plot.set_title(f"Z-Score Distribution: {region}", fontsize=14)
            ax_plot.set_xlabel("Z-Score (Deviation from Healthy Control Baseline)", fontsize=12)
            ax_plot.set_ylabel("Count (Number of Patients)", fontsize=12)
            ax_plot.legend()

            # Stats Text
            ax_text = plt.subplot(gs[1])
            ax_text.axis('off')

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

            ax_text.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                         verticalalignment='center', transform=ax_text.transAxes,
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Success! Report saved to: {os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    generate_report(INPUT_CSV, OUTPUT_PDF)