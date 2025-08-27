# Master Plotting Script with All Required Plots
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src_python'))

from src_python.data_loading import DataLoading
from src_python.plotting import *
from src_python.plotting import generate_figure_3_alternative_format

from src_python.statistical_analysis import StatisticalAnalysis
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """Generate all required plots as specified in Plots_list.txt."""

    print("="*80)
    print("ALP ANARCHY COMPREHENSIVE PLOTTING SUITE")
    print("="*80)
    print("Generating all required plots with enhanced statistical analysis...")

    # Initialize data loader
    data_loader = DataLoading()

    # Clear previous statistics file
    if STATS_FILE.exists():
        os.remove(STATS_FILE)

    # Initialize summary
    overall_summary = []

    # 0. Validate data integrity first
    print("\n[0] VALIDATING DATA INTEGRITY...")
    validation_results = data_loader.validate_data_integrity()

    print("\nData Validation Results:")
    print(tabulate(validation_results, headers=["File", "Status", "Rows"], tablefmt="grid"))

    failed_files = [r for r in validation_results if "FAIL" in r[1]]
    if failed_files:
        print(f"\nWARNING: {len(failed_files)} critical files are missing or corrupted!")
        print("Some plots may fail. Please ensure the C++ simulation has completed successfully.")

    # Debug available files
    data_loader.debug_available_files()

    # 1. FIGURE 2 MAIN ANALYSIS PLOTS
    print("\n[1] GENERATING FIGURE 2 MAIN ANALYSIS PLOTS...")
    N_values_fig2 = [2, 10, 30]  # Generate for multiple N values

    for N in N_values_fig2:
        try:
            print(f"\n  → Generating Figure 2 plots for N={N}...")
            generate_figure_2(N, data_loader)
            overall_summary.append([f"Figure 2 (N={N})", "Generated", "Multiple heatmaps with statistics"])
        except Exception as e:
            print(f"ERROR: Figure 2 (N={N}) failed: {str(e)}")
            overall_summary.append([f"Figure 2 (N={N})", "Failed", str(e)])

    print("\n[1a] GENERATING CHI-SQUARE HEATMAPS...")
    for N in N_values_fig2:
        try:
            print(f"\n → Generating Chi-Square heatmaps for N={N}...")
            generate_chi_square_heatmaps(N, data_loader)
            overall_summary.append([f"Chi-Square Heatmap (N={N})", "Generated", "Statistical validation plots"])
        except Exception as e:
            print(f"ERROR: Chi-Square heatmaps (N={N}) failed: {str(e)}")
            overall_summary.append([f"Chi-Square Heatmap (N={N})", "Failed", str(e)])

    # 2. FIGURE 3 MAIN ANALYSIS PLOTS
    print("\n[2] GENERATING FIGURE 3 MAIN ANALYSIS PLOTS...")
    try:
        print("  → Generating Figure 3 scaling analysis...")
        generate_figure_3(data_loader)
        overall_summary.append(["Figure 3", "Generated", "Scaling law with fit analysis"])
    except Exception as e:
        print(f"ERROR: Figure 3 failed: {str(e)}")
        overall_summary.append(["Figure 3", "Failed", str(e)])


    try:
        print(" → Generating Figure 3 exponential format...")
        generate_figure_3_alternative_format(data_loader)
        overall_summary.append(["Figure 3 Alt Format", "Generated", "Exponential format plot"])
    except Exception as e:
        print(f"ERROR: Figure 3 alternative format failed: {str(e)}")
        overall_summary.append(["Figure 3 Alt Format", "Failed", str(e)])


    # 2b. New Individual Log-Log Plot
    try:
        print(" → Generating Figure 3 log-log individual plot...")
        generate_figure_3_loglog(data_loader)
        overall_summary.append(["Figure 3 Log-Log Individual", "Generated", "Log-log scaling with fit and theory"])
    except Exception as e:
        print(f"ERROR: Figure 3 log-log individual failed: {str(e)}")
        overall_summary.append(["Figure 3 Log-Log Individual", "Failed", str(e)])



    # 3. STATISTICAL VALIDATION PLOTS
    print("\n[3] GENERATING STATISTICAL VALIDATION PLOTS...")

    # 3a. Probability and Matrix Element Distributions
    try:
        print("  → Generating diagnostic distributions...")
        plot_diagnostic_distributions(N_values_fig2, data_loader)
        overall_summary.append(["Diagnostic Distributions", "Generated", "Probability & matrix element analysis"])
    except Exception as e:
        print(f"ERROR: Diagnostic distributions failed: {str(e)}")
        overall_summary.append(["Diagnostic Distributions", "Failed", str(e)])

    # 3a-1. Conversion Histograms
    try:
        print("\n =========================================")
        print("  → Generating conversion histograms...")
        print("=========================================\n")
        plot_conversion_histograms(N_values_fig2, data_loader)
        overall_summary.append(["Conversion Histograms", "Generated", "Fitted conversion distributions"])
    except Exception as e:
        print(f"ERROR: Conversion histograms failed: {str(e)}")
        overall_summary.append(["Conversion Histograms", "Failed", str(e)])

    # 3b. CDF Plots and Statistical Tests
    try:
        print("\n ====================================================")
        print("  → Generating CDF plots and statistical tests...")
        print("===================================================== \n")
        plot_cdf_and_statistical_tests(N_values_fig2, data_loader)
        overall_summary.append(["CDF & Statistical Tests", "Generated", "Comprehensive distribution testing"])
    except Exception as e:
        print(f"ERROR: CDF plots failed: {str(e)}")
        overall_summary.append(["CDF & Statistical Tests", "Failed", str(e)])

    # 3c. Combined CDF Plot
    try:
        print("\n =========================================")
        print("  → Generating combined CDF plot...")
        print("=========================================")
        plot_combined_cdf(N_values_fig2, data_loader)
        overall_summary.append(["Combined CDF Plot", "Generated", "Multi-N CDF comparison"])
    except Exception as e:
        print(f"ERROR: Combined CDF plot failed: {str(e)}")
        overall_summary.append(["Combined CDF Plot", "Failed", str(e)])

    # 3d. Figure 3 CDF Plots (missing before)
    try:
        print("=========================================")
        print(" → Generating Figure 3 CDF plots...")
        print("=========================================")
        plot_figure3_cdf(data_loader)
        overall_summary.append(["Figure 3 CDF Plots", "Generated", "Survival probability CDFs for multiple N"])
    except Exception as e:
        print(f"ERROR: Figure 3 CDF plots failed: {str(e)}")
        overall_summary.append(["Figure 3 CDF Plots", "Failed", str(e)])


    # 4. DIAGNOSTIC & CONVERGENCE PLOTS
    print("\n[4] GENERATING DIAGNOSTIC & CONVERGENCE PLOTS...")

    # 4a. Convergence Analysis
    try:
        print("\n-------------------------------------------")
        print("  → Generating convergence analysis...")
        print("----------------------------------------------- \n")
        plot_convergence_and_diagnostics(data_loader)
        overall_summary.append(["Convergence Analysis", "Generated", "Multi-N convergence validation"])
    except Exception as e:
        print(f"ERROR: Convergence analysis failed: {str(e)}")
        overall_summary.append(["Convergence Analysis", "Failed", str(e)])

    # 4b. Flux Diagnostic Plots
    try:
        print("\n-------------------------------------------")
        print("  → Generating flux diagnostic plots...")
        print("----------------------------------------------- \n")
        plot_flux_comparisons(data_loader)
        overall_summary.append(["Flux Diagnostics", "Generated", "2D/3D flux analysis with fits"])
    except Exception as e:
        print(f"ERROR: Flux diagnostics failed: {str(e)}")
        overall_summary.append(["Flux Diagnostics", "Failed", str(e)])

    # 5. ADDITIONAL REQUIRED PLOTS (from Plots_list.txt)
    print("\n[5] GENERATING ADDITIONAL REQUIRED PLOTS...")

    # 5a. Realization Diagnostic Scatter Plot (Pe→γ vs Pγ→γ) - THIS WAS MISSING!
    try:
        print("  → Generating realization diagnostic scatter plots...")
        plot_realization_diagnostics(N_values_fig2, data_loader)
        overall_summary.append(["Realization Diagnostics", "Generated", "Conversion vs survival scatter plots"])
    except Exception as e:
        print(f"ERROR: Realization diagnostics failed: {str(e)}")
        overall_summary.append(["Realization Diagnostics", "Failed", str(e)])

    # 6. MASTER SUMMARY PLOT
    print("\n[6] GENERATING MASTER SUMMARY PLOT...")
    try:
        print("  → Generating master summary plot...")
        summary_result = generate_master_summary_plot(data_loader)
        overall_summary.extend(summary_result)
    except Exception as e:
        print(f"ERROR: Master summary plot failed: {str(e)}")
        overall_summary.append(["Master Summary", "Failed", str(e)])

    # FINAL SUMMARY
    print("\n" + "="*80)
    print("COMPREHENSIVE PLOTTING SUMMARY")
    print("="*80)

    print(tabulate(overall_summary, headers=["Plot Category", "Status", "Description"], tablefmt="grid"))

    # Count successes and failures
    generated = len([s for s in overall_summary if s[1] == "Generated"])
    failed = len([s for s in overall_summary if s[1] == "Failed"])

    print(f"\nSUMMARY STATISTICS:")
    print(f"  → Successfully generated: {generated}")
    print(f"  → Failed: {failed}")
    print(f"  → Success rate: {generated/(generated+failed)*100:.1f}%")

    if failed > 0:
        print(f"\nFAILED PLOTS:")
        for summary in overall_summary:
            if summary[1] == "Failed":
                print(f"  ✗ {summary[0]}: {summary[2]}")

        print(f"\nTROUBLESHOoting HINTS:")
        print(f"  1. Ensure the C++ simulation has completed successfully")
        print(f"  2. Check that all data files exist in the correct directories")
        print(f"  3. Verify data file formats match expected column counts")
        print(f"  4. Check the console output above for specific error messages")

    print(f"\nPlots saved to: {data_loader.PLOTS_DIR}")
    print(f"Statistics summary saved to: {STATS_FILE}")

    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)

def plot_realization_diagnostics(N_values, data_loader):
    """Generate realization diagnostic scatter plots (Pe→γ vs Pγ→γ) - MISSING FROM ORIGINAL!"""
    for N in N_values:
        try:
            # Load probability data
            probs = data_loader.load_precomputed_probabilities(N, figure_type='fig2')
            p_gg = probs[:, 0]  # Survival probability
            p_eg = probs[:, 1]  # Conversion probability

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Main scatter plot: Conversion vs Survival
            ax1.scatter(p_gg, p_eg, alpha=0.6, s=1, c='purple', rasterized=True)
            ax1.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival Probability)', fontsize=12)
            ax1.set_ylabel(r'$P_{e \to \gamma}$ (Conversion Probability)', fontsize=12)
            ax1.set_title(f'Conversion vs Survival Probability (N={N})', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Add expected value lines
            expected_p_gg = 0.625 if N == 2 else 1.0/N
            expected_p_eg = 0.25 if N == 2 else 0.1
            ax1.axvline(expected_p_gg, color='red', linestyle='--', alpha=0.8,
                       label=f'Expected P_gg ({expected_p_gg:.3f})')
            ax1.axhline(expected_p_eg, color='blue', linestyle='--', alpha=0.8,
                       label=f'Expected P_eg ({expected_p_eg:.3f})')
            ax1.legend(fontsize=10)

            # Calculate and display correlation
            correlation = np.corrcoef(p_gg, p_eg)[0, 1]
            corr_stats = {
                'Correlation': f'{correlation:.4f}',
                'Data Points': len(p_gg),
                'Independent?': 'Yes' if abs(correlation) < 0.1 else 'No'
            }
            add_statistics_textbox(ax1, corr_stats, 'upper right')

            # 2. Hexbin plot for density visualization
            hb = ax2.hexbin(p_gg, p_eg, gridsize=30, cmap='Blues', mincnt=1)
            ax2.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival Probability)', fontsize=12)
            ax2.set_ylabel(r'$P_{e \to \gamma}$ (Conversion Probability)', fontsize=12)
            ax2.set_title(f'Density Plot: Conversion vs Survival (N={N})', fontsize=14)
            plt.colorbar(hb, ax=ax2, label='Count')

            # 3. Marginal distribution: Survival probability
            ax3.hist(p_gg, bins=50, density=True, alpha=0.7, color='lightblue',
                    edgecolor='blue', label='Observed')
            ax3.axvline(expected_p_gg, color='red', linestyle='--', linewidth=2,
                       label=f'Expected ({expected_p_gg:.3f})')
            ax3.axvline(np.mean(p_gg), color='green', linestyle='-.', linewidth=2,
                       label=f'Observed Mean ({np.mean(p_gg):.3f})')
            ax3.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival Probability)', fontsize=12)
            ax3.set_ylabel('Probability Density', fontsize=12)
            ax3.set_title(f'Marginal Distribution: Survival (N={N})', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

            # 4. Marginal distribution: Conversion probability
            ax4.hist(p_eg, bins=50, density=True, alpha=0.7, color='lightgreen',
                    edgecolor='green', label='Observed')
            ax4.axvline(expected_p_eg, color='red', linestyle='--', linewidth=2,
                       label=f'Expected ({expected_p_eg:.3f})')
            ax4.axvline(np.mean(p_eg), color='blue', linestyle='-.', linewidth=2,
                       label=f'Observed Mean ({np.mean(p_eg):.3f})')
            ax4.set_xlabel(r'$P_{e \to \gamma}$ (Conversion Probability)', fontsize=12)
            ax4.set_ylabel('Probability Density', fontsize=12)
            ax4.set_title(f'Marginal Distribution: Conversion (N={N})', fontsize=12)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = data_loader.PLOTS_DIR / f'realization_diagnostics_N{N}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"ERROR: Realization diagnostics failed for N={N}: {str(e)}")
            plt.close()

if __name__ == "__main__":
    main()
