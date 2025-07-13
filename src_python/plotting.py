import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from .constants import CONSTANTS
from .statistical_analysis import StatisticalAnalysis
from tqdm import tqdm
import os
from tabulate import tabulate
from pathlib import Path

# Debug: Print plotting.py path
print(f"DEBUG: plotting.py loaded from: {os.path.abspath(__file__)}")

def generate_figure_2(N, data_loader):
    print(f"Generating Figure 2 plots for N={N}...")
    stat_analyzer = StatisticalAnalysis()
    try:
        data = data_loader.load_figure_2_data(N)
    except FileNotFoundError as e:
        print(f"Diagnostic: {str(e)}")
        return
    g_e_range = data['g_e_range']
    g_gamma_range = data['g_gamma_range']
    W = data['W'].T  # Transpose to match CAST3.py orientation
    p_gg_vals = data['p_gg_vals']
    p_eg_vals = data['p_eg_vals']
    phi_osc_vals = data['phi_osc_vals']
    u_i0_gamma_vals = data['u_i0_gamma_vals']
    u_i0_e_vals = data['u_i0_e_vals']
    os.makedirs(data_loader.PLOTS_DIR, exist_ok=True)

    plot_summary = []

    # Main W heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(g_e_range, g_gamma_range, W, cmap='viridis', vmin=0, vmax=1, shading='gouraud')
    plt.colorbar(label='W (Probability)')
    plt.xlabel('log10(g_e)')
    plt.ylabel('log10(g_gamma)')
    plt.title(f'Figure 2: W for N={N}')
    plt.grid(True, alpha=0.3)
    g_e_limit = np.log10(CONSTANTS['G_GAMMA_G_E_LIMIT'] / np.power(10, g_gamma_range))
    plt.plot(g_e_limit, g_gamma_range, 'r--', label='CAST Limit')
    plt.legend()
    output_file = data_loader.PLOTS_DIR / f'figure_2_main_N{N}.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print(f"Saved Figure 2 main plot for N={N}")
    plot_summary.append(["W Heatmap", "Saved", f"figure_2_main_N{N}.png"])

    # Residual heatmap
    residuals = stat_analyzer.compute_residuals(W, g_e_range, g_gamma_range)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(g_e_range, g_gamma_range, residuals, cmap='coolwarm', vmin=-0.5, vmax=0.5, shading='gouraud')
    plt.colorbar(label='Residuals (W - Expected)')
    plt.xlabel('log10(g_e)')
    plt.ylabel('log10(g_gamma)')
    plt.title(f'Figure 2: Residuals for N={N}')
    plt.grid(True, alpha=0.3)
    output_file = data_loader.PLOTS_DIR / f'figure_2_residuals_N{N}.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print(f"Saved Figure 2 residuals plot for N={N}")
    plot_summary.append(["Residuals Heatmap", "Saved", f"figure_2_residuals_N{N}.png"])

    # Chi-squared p-value heatmap
    chi2_p_vals = stat_analyzer.compute_chi2_p_values(W, N, g_e_range, g_gamma_range)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(g_e_range, g_gamma_range, chi2_p_vals, cmap='viridis', vmin=0, vmax=1, shading='gouraud')
    plt.colorbar(label='Chi-squared p-value')
    plt.xlabel('log10(g_e)')
    plt.ylabel('log10(g_gamma)')
    plt.title(f'Figure 2: Chi-squared p-values for N={N}')
    plt.grid(True, alpha=0.3)
    output_file = data_loader.PLOTS_DIR / f'figure_2_chi2_N{N}.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print(f"Saved Figure 2 chi-squared plot for N={N}")
    plot_summary.append(["Chi-squared Heatmap", "Saved", f"figure_2_chi2_N{N}.png"])

    # Q-Q plot for p_gg
    plt.figure(figsize=(8, 6))
    probplot(p_gg_vals.flatten(), dist='uniform' if N == 2 else 'norm', sparams=(np.mean(p_gg_vals), np.std(p_gg_vals)), plot=plt)
    plt.title(f'Q-Q Plot for p_gg (N={N})')
    output_file = data_loader.PLOTS_DIR / f'figure_2_p_gg_qq_N{N}.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print(f"Saved Figure 2 p_gg Q-Q plot for N={N}")
    plot_summary.append(["p_gg Q-Q Plot", "Saved", f"figure_2_p_gg_qq_N{N}.png"])

    # Residual histogram
    plt.figure(figsize=(8, 6))
    plt.hist(residuals.flatten(), bins=50, density=True, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title(f'Figure 2: Residuals Histogram (N={N})')
    plt.grid(True, alpha=0.3)
    output_file = data_loader.PLOTS_DIR / f'figure_2_residuals_hist_N{N}.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print(f"Saved Figure 2 residuals histogram for N={N}")
    plot_summary.append(["Residuals Histogram", "Saved", f"figure_2_residuals_hist_N{N}.png"])

    # Statistical tests
    stat_pgg, p_value_pgg = stat_analyzer.kstest(p_gg_vals.flatten(), 'uniform' if N == 2 else 'norm',
                                                 args=() if N == 2 else (np.mean(p_gg_vals), np.std(p_gg_vals)))
    stat_peg, p_value_peg = stat_analyzer.kstest(p_eg_vals.flatten(), 'uniform' if N == 2 else 'norm',
                                                 args=() if N == 2 else (np.mean(p_eg_vals), np.std(p_eg_vals)))
    stat_u_i0, p_value_u_i0 = stat_analyzer.kstest(u_i0_gamma_vals.flatten(), 'norm',
                                                   args=(np.mean(u_i0_gamma_vals), np.std(u_i0_gamma_vals)))
    stat_W, p_value_W = stat_analyzer.kstest(W.flatten(), 'uniform', args=(0, 1))

    table = [
        ["Mean Chi^2 p-value", f"{np.mean(chi2_p_vals):.6f}", "Yes" if np.mean(chi2_p_vals) < 0.05 else "No"],
        [r"P_{\gamma\to\gamma} KS p-value", f"{p_value_pgg:.4f}", "Yes" if p_value_pgg < 0.05 else "No"],
        [r"P_{e\to\gamma} KS p-value", f"{p_value_peg:.4f}", "Yes" if p_value_peg < 0.05 else "No"],
        ["U_{1,0} KS p-value", f"{p_value_u_i0:.4f}", "Yes" if p_value_u_i0 < 0.05 else "No"],
        ["W Uniformity KS p-value", f"{p_value_W:.6f}", "Yes" if p_value_W < 0.05 else "No"]
    ]
    print(f"\nStatistical Tests for Figure 2 (N={N}):")
    print(tabulate(table, headers=["Metric", "p-value", "Significant?"], tablefmt="grid"))
    plot_summary.append(["Statistical Tests", "Completed", "Printed to console"])

    print("\nFigure 2 Plot Summary (N={}):".format(N))
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))

def generate_figure_3(data_loader):
    print("Generating Figure 3 plots...")
    stat_analyzer = StatisticalAnalysis()
    try:
        data = data_loader.load_figure_3_data()
    except FileNotFoundError as e:
        print(f"Diagnostic: {str(e)}")
        return
    N_range = data['N_range']
    log_g_50 = data['log_g_50']
    W_vals = data['W_vals']
    plot_summary = []

    slope, c, fit_log_g, residuals, r_squared = stat_analyzer.fit_figure_3(N_range, log_g_50)

    # Main plot
    plt.figure(figsize=(10, 8))
    plt.plot(N_range, log_g_50, 'bo', label='Data')
    plt.plot(N_range, fit_log_g, 'r-', label=f'Fit: slope={slope:.3f}, c={c:.3f}, R^2={r_squared:.3f}')
    plt.xlabel('N')
    plt.ylabel('log10(g_gamma)')
    plt.title('Figure 3: log10(g_gamma) at W=0.5 vs N')
    plt.grid(True)
    plt.legend()
    output_file = data_loader.PLOTS_DIR / 'figure_3.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print("Saved Figure 3 main plot")
    plot_summary.append(["Main Plot", "Saved", "figure_3.png"])

    # Residual histogram
    plt.figure(figsize=(8, 6))
    plt.hist(residuals.flatten(), bins=50, density=True, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Figure 3: Residuals Histogram')
    plt.grid(True, alpha=0.3)
    output_file = data_loader.PLOTS_DIR / 'figure_3_residuals_hist.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print("Saved Figure 3 residuals histogram")
    plot_summary.append(["Residuals Histogram", "Saved", "figure_3_residuals_hist.png"])

    # Q-Q plot for residuals
    plt.figure(figsize=(8, 6))
    probplot(residuals.flatten(), dist='norm', plot=plt)
    plt.title('Q-Q Plot for Figure 3 Residuals')
    output_file = data_loader.PLOTS_DIR / 'figure_3_residuals_qq.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print("Saved Figure 3 residuals Q-Q plot")
    plot_summary.append(["Residuals Q-Q Plot", "Saved", "figure_3_residuals_qq.png"])

    print("\nFigure 3 Plot Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))

def plot_distributions(N_values, data_loader):
    plot_summary = []
    for N in tqdm(N_values, desc="Generating distribution plots"):
        try:
            data = data_loader.load_matrix_distributions(N)
        except FileNotFoundError as e:
            print(f"Diagnostic: {str(e)}")
            continue
        elements, determinants = data
        data_fig2 = data_loader.load_figure_2_data(N)
        p_gg_vals = data_fig2['p_gg_vals']
        p_eg_vals = data_fig2['p_eg_vals']
        u_i0_gamma_vals = data_fig2['u_i0_gamma_vals']

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.hist(p_gg_vals.flatten(), bins=50, density=True, alpha=0.7, label='p_gg')
        plt.xlabel('p_gg')
        plt.ylabel('Density')
        plt.title(f'p_gg Distribution (N={N})')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.hist(p_eg_vals.flatten(), bins=50, density=True, alpha=0.7, label='p_eg')
        plt.xlabel('p_eg')
        plt.ylabel('Density')
        plt.title(f'p_eg Distribution (N={N})')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.hist(u_i0_gamma_vals.flatten(), bins=50, density=True, alpha=0.7, label='u_i0_gamma')
        plt.xlabel('u_i0_gamma')
        plt.ylabel('Density')
        plt.title(f'u_i0_gamma Distribution (N={N})')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.hist(elements.flatten(), bins=50, density=True, alpha=0.7, label='Matrix Elements')
        plt.xlabel('Element Value')
        plt.ylabel('Density')
        plt.title(f'Matrix Elements Distribution (N={N})')
        plt.legend()

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / f'distributions_N{N}.png'
        plt.savefig(output_file)
        print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
        plt.close()
        print(f"Saved distribution plots for N={N}")
        plot_summary.append([f"Distributions (N={N})", "Saved", f"distributions_N{N}.png"])

    print("\nDistributions Plot Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))

def plot_diagnostic_distributions_fig2(N_values, data_loader):
    plot_summary = []
    for N in tqdm(N_values, desc="Generating diagnostic distributions for Figure 2"):
        try:
            elements, determinants = data_loader.load_matrix_distributions(N)
        except FileNotFoundError as e:
            print(f"Diagnostic: {str(e)}")
            continue
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(elements.flatten(), bins=50, density=True, alpha=0.7, label='Matrix Elements')
        plt.xlabel('Element Value')
        plt.ylabel('Density')
        plt.title(f'Matrix Elements Distribution (N={N})')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(determinants.flatten(), bins=50, density=True, alpha=0.7, label='Determinants')
        plt.xlabel('Determinant')
        plt.ylabel('Density')
        plt.title(f'Matrix Determinants Distribution (N={N})')
        plt.legend()

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / f'matrix_distributions_N{N}.png'
        plt.savefig(output_file)
        print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
        plt.close()
        print(f"Saved matrix distribution plots for N={N}")
        plot_summary.append([f"Matrix Distributions (N={N})", "Saved", f"matrix_distributions_N{N}.png"])

    print("\nDiagnostic Fig2 Plot Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))

def plot_diagnostic_distributions_fig3(data_loader):
    print("Generating diagnostic distributions for Figure 3...")
    plot_summary = []
    try:
        data = data_loader.load_diagnostics_flux_data()
    except FileNotFoundError as e:
        print(f"Diagnostic: {str(e)}")
        return
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(data[1].flatten(), bins=50, density=True, alpha=0.7, label='phi_osc')
    plt.xlabel('phi_osc')
    plt.ylabel('Density')
    plt.title('phi_osc Distribution')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(data[2].flatten(), bins=50, density=True, alpha=0.7, label='phi_b')
    plt.xlabel('phi_b')
    plt.ylabel('Density')
    plt.title('phi_b Distribution')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(data[3].flatten(), bins=50, density=True, alpha=0.7, label='phi_c')
    plt.xlabel('phi_c')
    plt.ylabel('Density')
    plt.title('phi_c Distribution')
    plt.legend()

    plt.tight_layout()
    output_file = data_loader.PLOTS_DIR / 'diagnostic_distributions_fig3.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print("Saved Figure 3 diagnostic distributions")
    plot_summary.append(["Diagnostic Fig3", "Saved", "diagnostic_distributions_fig3.png"])

    print("\nDiagnostic Fig3 Plot Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))

def generate_convergence_plots(data_loader):
    print("Generating convergence plots...")
    plot_summary = []
    try:
        fig2_data = data_loader.load_convergence_fig2_data()
        fig3_data = data_loader.load_convergence_fig3_data()
    except FileNotFoundError as e:
        print(f"Diagnostic: {str(e)}")
        return

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fig2_data[0], fig2_data[1], 'bo-', label='W')
    plt.xlabel('N')
    plt.ylabel('W')
    plt.title('Convergence Figure 2: W vs N')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fig3_data[0], fig3_data[1], 'ro-', label='W')
    plt.xlabel('N')
    plt.ylabel('W')
    plt.title('Convergence Figure 3: W vs N')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    output_file = data_loader.PLOTS_DIR / 'convergence_plots.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print("Saved convergence plots")
    plot_summary.append(["Convergence Plots", "Saved", "convergence_plots.png"])

    print("\nConvergence Plot Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))

def diagnose_figure_3(data_loader):
    print("Generating Figure 3 diagnostics...")
    plot_summary = []
    try:
        data = data_loader.load_diagnostics_realization_data()
    except FileNotFoundError as e:
        print(f"Diagnostic: {str(e)}")
        return
    plt.figure(figsize=(10, 8))
    plt.plot(data[0], data[2], 'bo-', label='SEM')
    plt.xlabel('N')
    plt.ylabel('SEM of p_gg')
    plt.title('Diagnostics: SEM of p_gg vs N')
    plt.grid(True)
    plt.legend()
    output_file = data_loader.PLOTS_DIR / 'diagnostics_realization.png'
    plt.savefig(output_file)
    print(f"DEBUG: Saving plot to: {os.path.abspath(output_file)}")
    plt.close()
    print("Saved Figure 3 diagnostics plot")
    plot_summary.append(["Figure 3 Diagnostics", "Saved", "diagnostics_realization.png"])

    print("\nFigure 3 Diagnostics Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
