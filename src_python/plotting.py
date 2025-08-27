# Enhanced Plotting Module with Comprehensive Statistical Analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import arcsine, beta, linregress, expon, norm, kstest, chi2_contingency
from scipy.optimize import curve_fit
from scipy.ndimage import label
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from constants import CONSTANTS
from statistical_analysis import StatisticalAnalysis
from tabulate import tabulate
import os
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import warnings

# Enable Matplotlib's mathtext for LaTeX rendering
plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=12)
plt.rc('mathtext', fontset='cm')


BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / 'plots'
# Statistics summary file
STATS_FILE = PLOTS_DIR / 'statistics_summary.txt'

def save_stats_to_file(table, headers, title):
    """Save a statistical table to the statistics summary file."""
    os.makedirs(STATS_FILE.parent, exist_ok=True)
    with open(STATS_FILE, 'a') as f:
        f.write(f"\n{title}\n")
        f.write("="*len(title) + "\n")
        f.write(tabulate(table, headers=headers, tablefmt="grid"))
        f.write("\n")
    print(f"DEBUG: Saved statistics table to: {os.path.abspath(STATS_FILE)}")

def g_gamma_n1(g_e):
    """Compute single-ALP CAST bound as in CAST4.py."""
    function_value = np.minimum(CONSTANTS['G_GAMMA_N1'], CONSTANTS['G_GAMMA_G_E_LIMIT'] / g_e)
    return function_value

def add_statistics_textbox(ax, stats_dict, location='upper right'):
    """Add a statistics textbox to a plot."""
    stats_text = []
    for key, value in stats_dict.items():
        if isinstance(value, float):
            if abs(value) < 0.001 or abs(value) > 1000:
                stats_text.append(f"{key}: {value:.3e}")
            else:
                stats_text.append(f"{key}: {value:.3f}")
        else:
            stats_text.append(f"{key}: {value}")

    textstr = '\n'.join(stats_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02 if 'left' in location else 0.98,
            0.98 if 'upper' in location else 0.02,
            textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top' if 'upper' in location else 'bottom',
            horizontalalignment='left' if 'left' in location else 'right',
            bbox=props)

def generate_figure_2(N, data_loader):
    """Generate enhanced Figure 2 plots with comprehensive statistical analysis."""
    print(f"Generating Figure 2 plots for N={N}...")
    stat_analyzer = StatisticalAnalysis()

    try:
        data = data_loader.load_figure_2_data(N)
    except Exception as e:
        print(f"ERROR: Failed to load data for Figure 2 (N={N}): {str(e)}")
        return

    g_e_range = data['g_e_range']
    g_gamma_range = data['g_gamma_range']

    # FIX: Don't transpose W here - keep original orientation
    W = data['W']
    P_gg_mean = data['P_gg_mean']
    P_gg_std = data['P_gg_std']
    P_eg_mean = data['P_eg_mean']
    P_eg_std = data['P_eg_std']
    W_residual_mean = data['W_residual_mean']
    # Find zero-residual contour once, reuse on all sub-plots
    zero_resid = W_residual_mean == 0
    zero_mask = W_residual_mean == 0
    labels, nlab = label(zero_mask)
    sizes = np.bincount(labels.ravel())
    if len(sizes) > 1:  # If there are components
        largest = np.argmax(sizes[1:]) + 1  # skip background=0
        outlier_mask = (labels != 0) & (labels != largest)
        W_masked = W.copy()
        W_masked[outlier_mask] = np.nan  # Mask outliers as NaN (won't be colored)
    else:
        W_masked = W  # No masking if no components

    W_chi2_p_value = data['W_chi2_p_value']

    # Compute CAST line
    g_n1_vals = np.array([g_gamma_n1(10**ge) for ge in g_e_range])
    cast_line_y = np.log10(g_n1_vals)

    os.makedirs(data_loader.PLOTS_DIR, exist_ok=True)
    plot_summary = []

    # Enhanced statistics for the dataset
    W_stats = {
        'Mean W': np.mean(W),
        'Std W': np.std(W),
        'Grid Points': W.size,
        'N Realizations': CONSTANTS.get('N_REALIZATIONS', 'Unknown')
    }

    # 1. Main Fraction of Realizations Heatmap
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        # FIX: Use .T to transpose for proper display orientation
        im = ax.pcolormesh(g_e_range, g_gamma_range, W_masked.T, cmap='viridis',
                           vmin=0, vmax=1, shading='gouraud')

        # Add colorbar with enhanced labeling
        cbar = plt.colorbar(im, label='W (Fraction of Successful Realizations)')
        cbar.ax.tick_params(labelsize=10)

        # Plot CAST constraint line
        ax.plot(g_e_range, cast_line_y, 'r-', linewidth=3,label='CAST Constraint (Single ALP)', alpha=0.8)


        # Enhanced labeling
        ax.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=14)
        ax.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=14)
        ax.set_title(f'Fraction of Realizations Consistent with CAST (N={N})', fontsize=16)
        # ── plot relaxed-bound (=0 residual) line ─────────
        # where residual grid crosses zero; use contour for robustness
        # ── plot relaxed-bound (=0 residual) continuous line ─────────
        # where residual grid crosses zero; use contour for robustness
        try:
            zr = ax.contour(g_e_range, g_gamma_range, W_residual_mean.T,
                            levels=[0], colors='k', linewidths=2,
                            linestyles='-')  # Changed to solid continuous line

            # Find the longest continuous path (main separator, ignore outliers)
            main_verts = None
            max_length = 0
            for collection in zr.collections:
                for path in collection.get_paths():
                    verts = path.vertices
                    if len(verts) > max_length:
                        max_length = len(verts)
                        main_verts = verts

            if main_verts is not None:
                # Plot only the main continuous line
                ax.plot(main_verts[:, 0], main_verts[:, 1], color='k', linewidth=2,
                        linestyle='-', label='Relaxed Bound (0-Residual)')

                # Save only the main line to file
                out = np.column_stack([main_verts[:, 0], main_verts[:, 1]])
                header = ("# Main zero-residual separator line for Figure-2 main heatmap\n"
                          "# (Longest continuous contour segment; outliers ignored)\n"
                          "# log10(g_e)   log10(g_gamma)\n")
                np.savetxt(data_loader.PLOTS_DIR / f'Relaxed_Bound_N{N}.txt',
                           out, header=header, fmt='%.6e')
        except Exception:
            pass
        # --------------------------------------------------



        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics textbox
        add_statistics_textbox(ax, W_stats, 'upper left')

        output_file = data_loader.PLOTS_DIR / f'figure_2_main_heatmap_N{N}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Main Fraction Heatmap", "Saved", output_file.name])

    except Exception as e:
        plot_summary.append(["Main Fraction Heatmap", "Failed", str(e)])
        plt.close()

    # 2. Statistical Validation: Residuals Heatmap
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        # FIX: Use .T to transpose for proper display orientation
        im = ax.pcolormesh(g_e_range, g_gamma_range, W_residual_mean.T,
                          cmap='RdBu_r', vmin=-0.5, vmax=0.5, shading='gouraud')
        cbar = plt.colorbar(im, label='Residuals (W - 0.5)')
        cbar.ax.tick_params(labelsize=10)

        ax.plot(g_e_range, cast_line_y, 'k--', linewidth=3,
               label='CAST Constraint', alpha=0.8)

        # Add zero-residual contour
        try:
            contour = ax.contour(g_e_range, g_gamma_range, W_residual_mean.T,
                               levels=[0], colors='black', linewidths=2, alpha=0.7)
            ax.clabel(contour, inline=True, fontsize=10, fmt='Zero Residual')
        except:
            pass

        ax.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=14)
        ax.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=14)
        ax.set_title(f'Statistical Residuals Heatmap (N={N})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Residual statistics
        residual_stats = {
            'Mean Residual': np.mean(W_residual_mean),
            'RMS Residual': np.sqrt(np.mean(W_residual_mean**2)),
            'Max |Residual|': np.max(np.abs(W_residual_mean))
        }
        add_statistics_textbox(ax, residual_stats, 'upper left')

        output_file = data_loader.PLOTS_DIR / f'figure_2_residuals_heatmap_N{N}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Residuals Heatmap", "Saved", output_file.name])
    except Exception as e:
        plot_summary.append(["Residuals Heatmap", "Failed", str(e)])
        plt.close()

    # 3. Enhanced P_gg and P_eg Mean Heatmaps (Side by side)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # P_gg mean heatmap - FIX: Use .T for proper orientation
        im1 = ax1.pcolormesh(g_e_range, g_gamma_range, P_gg_mean.T,
                           cmap='plasma', shading='gouraud')
        cbar1 = plt.colorbar(im1, ax=ax1, label=r'$\langle P_{\gamma \to \gamma} \rangle$')
        ax1.plot(g_e_range, cast_line_y, 'k--', linewidth=2,
                label='CAST Constraint', alpha=0.8)
        ax1.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=14)
        ax1.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=14)
        ax1.set_title(f'Mean Survival Probability (N={N})', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # P_eg mean heatmap - FIX: Use .T for proper orientation
        im2 = ax2.pcolormesh(g_e_range, g_gamma_range, P_eg_mean.T,
                           cmap='viridis', shading='gouraud')
        cbar2 = plt.colorbar(im2, ax=ax2, label=r'$\langle P_{e \to \gamma} \rangle$')
        ax2.plot(g_e_range, cast_line_y, 'k--', linewidth=2,
                label='CAST Constraint', alpha=0.8)
        ax2.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=14)
        ax2.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=14)
        ax2.set_title(f'Mean Conversion Probability (N={N})', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add expected value information
        expected_p_gg = 0.625 if N == 2 else 1.0/N
        expected_p_eg = 0.25 if N == 2 else 0.1

        p_gg_stats = {
            'Expected': f'{expected_p_gg:.3f}',
            'Mean Observed': f'{np.mean(P_gg_mean):.3f}',
            'Std Dev': f'{np.mean(P_gg_std):.3f}'
        }

        p_eg_stats = {
            'Expected': f'{expected_p_eg:.3f}',
            'Mean Observed': f'{np.mean(P_eg_mean):.3f}',
            'Std Dev': f'{np.mean(P_eg_std):.3f}'
        }

        add_statistics_textbox(ax1, p_gg_stats, 'upper left')
        add_statistics_textbox(ax2, p_eg_stats, 'upper left')

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / f'figure_2_probabilities_N{N}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Probability Heatmaps", "Saved", output_file.name])
    except Exception as e:
        plot_summary.append(["Probability Heatmaps", "Failed", str(e)])
        plt.close()

    print(f"\nFigure 2 Summary (N={N}):")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
    save_stats_to_file(plot_summary, ["Plot", "Status", "Output File"], f"Figure 2 Summary (N={N})")


def generate_chi_square_heatmaps(N, data_loader):
    """Generate chi-square heatmaps similar to Mathematica code."""
    print(f"Generating Chi-Square heatmaps for N={N}...")

    try:
        data = data_loader.load_figure_2_data(N)
    except Exception as e:
        print(f"ERROR: Failed to load data for Chi-Square plots (N={N}): {str(e)}")
        return

    g_e_range = data['g_e_range']
    g_gamma_range = data['g_gamma_range']
    W = data['W']  # Don't transpose here

    # Compute chi-square matrix
    expected_fraction = np.mean(W)
    chi_square_matrix = np.where(expected_fraction > 0,
                                (W - expected_fraction)**2 / expected_fraction, 0)

    os.makedirs(data_loader.PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use proper orientation - transpose chi_square_matrix for display
    im = ax.pcolormesh(g_e_range, g_gamma_range, chi_square_matrix.T,
                       cmap='rainbow', shading='gouraud')

    cbar = plt.colorbar(im, label='Chi-Square Value')
    cbar.ax.tick_params(labelsize=10)

    # Add CAST constraint line
    g_n1_vals = np.array([g_gamma_n1(10**ge) for ge in g_e_range])
    cast_line_y = np.log10(g_n1_vals)
    ax.plot(g_e_range, cast_line_y, 'k-', linewidth=3, alpha=0.8, label='CAST Constraint')

    ax.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=14)
    ax.set_title(f'Chi-Square Statistic for N={N}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add statistics
    chi2_stats = {
        'Max Chi²': f'{np.max(chi_square_matrix):.3f}',
        'Mean Chi²': f'{np.mean(chi_square_matrix):.3f}',
        'Expected W': f'{expected_fraction:.3f}'
    }
    add_statistics_textbox(ax, chi2_stats, 'upper left')

    output_file = data_loader.PLOTS_DIR / f'chi_square_heatmap_N{N}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chi-square heatmap saved: {output_file.name}")

def power_law_func(N, m, c):
    return m * np.log10(N) - c / np.log(10)


def generate_figure_3(data_loader):
    """Generate enhanced Figure 3 plots with comprehensive fitting analysis."""
    print("Generating Figure 3 plots...")
    stat_analyzer = StatisticalAnalysis()



    try:
        data = data_loader.load_figure_3_data()
        fit_data = data_loader.load_fit_plot_data()
        power_law = data_loader.load_power_law_fit()
    except Exception as e:
        print(f"ERROR: Failed to load data for Figure 3: {str(e)}")
        return

    N_range = data['N_range']
    log_g_50 = data['log_g_50gamma']
    W = data['W'].T

    os.makedirs(data_loader.PLOTS_DIR, exist_ok=True)
    plot_summary = []

    # 1. Main Scaling Plot with Enhanced Fitting
    log_N = np.log10(N_range)

    # Fit using curve_fit, not C++-derived fit
    popt, pcov = curve_fit(power_law_func, N_range, log_g_50, p0=[0.25, 23])
    m_fit, c_fit = popt

    fit_log_g = power_law_func(N_range, m_fit, c_fit)
    # R^2 calculation
    residuals = log_g_50 - fit_log_g
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_g_50 - np.mean(log_g_50))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(N_range, log_g_50, 'bo', markersize=8, label='Simulation Data')
        ax.plot(N_range, fit_log_g, 'r-', linewidth=3, alpha=0.8,
            label=f'Power Law Fit: $y = m \\log_{{10}}(N) - c/\\ln(10)$')
        # Theoretical N^(1/4) line for comparison
        theoretical_slope = 0.25
        theoretical_fit = theoretical_slope * log_N - 23 / np.log(10)
        label_txt = r'Theory: $0.25\,\log_{{10}}(N) - 23/\ln(10)$'
        ax.plot(N_range, theoretical_fit, 'g--', linewidth=2, alpha=0.7,
            label=label_txt)

        ax.set_xlabel('Number of ALPs (N)', fontsize=14)
        ax.set_ylabel(r'$\log_{10}(g_{\gamma,50} / \mathrm{GeV}^{-1})$', fontsize=14)
        ax.set_title('CAST Bound on Photon Coupling vs. Number of ALPs', fontsize=16)
        ax.set_xscale('linear')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Enhanced statistics
        fit_stats = {
            'Fitted $m$': f'{m_fit:.4f}',
            'Fitted $c$': f'{c_fit:.4f}',
            '$R^2$': f'{r_squared:.4f}',
            'Fit equation': r'$y = m \log_{10}(N) - c/\ln(10)$',
            'Deviation': f'{abs(m_fit - 0.25):.4f}'
            }
        add_statistics_textbox(ax, fit_stats, 'upper right')

        output_file = data_loader.PLOTS_DIR / 'figure_3_main_scaling.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Main Scaling Plot", "Saved", output_file.name])

    except Exception as e:
        plot_summary.append(["Main Scaling Plot", "Failed", str(e)])
        plt.close()

# 2. Enhanced Residuals Analysis
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        residuals = fit_data['residual']

        # Residuals vs N - ADD legend
        ax1.plot(fit_data['N'], residuals, 'bo-', markersize=6, linewidth=2, label='Residuals')
        ax1.axhline(0, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Zero Line')
        ax1.set_xlabel('Number of ALPs (N)', fontsize=12)
        ax1.set_ylabel('Fit Residuals', fontsize=12)
        ax1.set_title('Power-Law Fit Residuals vs N', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='upper right')  # Added legend

        # Residual statistics
        residual_stats = {
            'Mean Residual': np.mean(residuals),
            'RMS Residual': np.sqrt(np.mean(residuals**2)),
            'Max |Residual|': np.max(np.abs(residuals))
        }
        add_statistics_textbox(ax1, residual_stats, 'upper left')  # Moved to left

        # Residuals histogram with normality test
        ax2.hist(residuals, bins=10, density=True, alpha=0.7,
             color='lightblue', edgecolor='blue', label='Residuals')
        # Fit normal distribution to residuals
        mu, sigma = norm.fit(residuals)
        x = np.linspace(np.min(residuals), np.max(residuals), 100)
        ax2.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
             label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
        # Kolmogorov-Smirnov test for normality
        ks_stat, p_value = kstest(residuals, 'norm', args=(mu, sigma))
        ax2.set_xlabel('Residual Value', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('Distribution of Fit Residuals', fontsize=14)
        ax2.legend(fontsize=10, loc='upper left')  # Moved to left
        ax2.grid(True, alpha=0.3)

        normality_stats = {
        'KS Statistic': f'{ks_stat:.4f}',
        'p-value': f'{p_value:.4f}',
        'Normal?': 'Yes' if p_value > 0.05 else 'No'
        }
        add_statistics_textbox(ax2, normality_stats, 'upper right')

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / 'figure_3_residuals_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Residuals Analysis", "Saved", output_file.name])

    except Exception as e:
        plot_summary.append(["Residuals Analysis", "Failed", str(e)])
        plt.close()


    print("\nFigure 3 Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
    save_stats_to_file(plot_summary, ["Plot", "Status", "Output File"], "Figure 3 Summary")


def generate_figure_3_alternative_format(data_loader):
    """Generate Figure 3 in the paper's format: log10(g_gamma) vs N with exponential fit."""
    print("Generating Figure 3 alternative format plot...")

    try:
        data = data_loader.load_figure_3_data()
        power_law = data_loader.load_power_law_fit()
    except Exception as e:
        print(f"ERROR: Failed to load data for Figure 3 alternative: {str(e)}")
        return

    N_range = data['N_range']
    N_fit = np.logspace(np.log10(min(N_range)), np.log10(max(N_range)), 200)

    log_g_50 = data['log_g_50gamma']

    os.makedirs(data_loader.PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot data points
    ax.plot(N_range, log_g_50, 'ro', markersize=8, label='Simulation Data',
            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)

    # Exponential fit line (from power law parameters)
    # Calculated fit (on raw data, no change)
    slope = power_law['m']
    c = power_law['c']
    log_g_fit = slope * np.log10(N_fit)-c
    ax.plot(N_fit, log_g_fit, 'b-', linewidth=3, alpha=0.8, label=rf'Exponential Fit: $g_{{\gamma,50}} \propto N^{{{slope:.3f}}}$')
    ax.set_xscale('linear')  # Add this line after ax creation

    # Theoretical line using theoretical parameters (fixed slope=0.25, intercept according to paper)
    theoretical_slope = 0.25
    # Fit intercept with fixed slope on raw log_N and log_g_50

    theoretical_intercept = -23/np.log(10)  # Use raw fitted intercept with theoretical slope
    theoretical_fit = theoretical_slope * np.log10(N_fit) + theoretical_intercept
    ax.plot(N_fit, theoretical_fit, 'g--', linewidth=3, alpha=0.8, label=rf'Theoretical: $g_{{\gamma,50}} \propto N^{{1/4}} \cdot 10^{{{theoretical_intercept:.3f}}}$')


    ax.set_xlabel('Number of ALPs (N)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(g_{\gamma,50} / \mathrm{GeV}^{-1})$', fontsize=14)
    ax.set_title('CAST Bound Scaling: Exponential Fit Format', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add fit statistics
    r_squared = power_law['r_squared']
    fit_stats = {
        'Fitted Exponent': f'{slope:.4f}',
        'Theoretical': '0.25',
        'R²': f'{r_squared:.4f}',
        'Deviation': f'{abs(slope - 0.25):.4f}'
    }
    add_statistics_textbox(ax, fit_stats, 'lower right')

    output_file = data_loader.PLOTS_DIR / 'figure_3_exponential_format.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure 3 exponential format saved: {output_file.name}")

def generate_figure_3_loglog(data_loader):
    """Generate individual log-log plot for Figure 3: log10(g_50,gamma) vs log10(N) with fit, theory, legend, and stats box."""
    try:
        data = data_loader.load_figure_3_data()
        N_range = data['N_range']
        log_g_50 = data['log_g_50gamma']

        log_N = np.log10(N_range)

        # Perform linear fit (calculated)
        slope, intercept, r_value, p_value, std_err = linregress(log_N, log_g_50)
        r_squared = r_value**2
        fit_line = slope * log_N + intercept

        # Theoretical line (slope=0.25, intercept derived consistently)
        theoretical_slope = 0.25
        c_fit = -intercept*np.log(10)
        theoretical_intercept = -(23/np.log(10))
        theoretical_fit = theoretical_slope * log_N + theoretical_intercept

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(log_N, log_g_50, 'bo', markersize=8, label='Simulation Data')
        ax.plot(log_N, fit_line, 'r-', linewidth=3, alpha=0.8,
                label=f'Fit: $y = {slope:.4f} \\log_{{10}}(N) + {intercept:.4f}$')
        ax.plot(log_N, theoretical_fit, 'g--', linewidth=2, alpha=0.7,
                label=f'Theory: $y = 0.25 \\log_{{10}}(N) + {theoretical_intercept:.4f}$')

        ax.set_xlabel(r'$\log_{10}(N)$ (Number of ALPs)', fontsize=14)
        ax.set_ylabel(r'$\log_{10}(g_{\gamma,50} / \mathrm{GeV}^{-1})$', fontsize=14)
        ax.set_title('Figure 3: Log-Log Scaling Law', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        fit_stats = {
            'Fitted Slope': f'{slope:.4f}',
            'Fitted Intercept': f'{intercept:.4f}',
            '$R^2$': f'{r_squared:.4f}',
            'Theoretical Slope': '0.25',
            'Theoretical Intercept': f'{theoretical_intercept:.4f}'
        }
        add_statistics_textbox(ax, fit_stats, 'upper right')

        output_file = data_loader.PLOTS_DIR / 'figure_3_loglog_scaling.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Figure 3 log-log plot saved: {output_file.name}")
        return ["Figure 3 Log-Log", "Saved", output_file.name]

    except Exception as e:
        return ["Figure 3 Log-Log", "Failed", str(e)]



def plot_diagnostic_distributions(N_values, data_loader):
    """Generate comprehensive diagnostic distribution plots with statistical validation."""
    stat_analyzer = StatisticalAnalysis()

    for N in N_values:
        plot_summary = []

        # Load probability data
        probs = None
        try:
            probs = data_loader.load_precomputed_probabilities(N, figure_type='fig2')
        except Exception as e1:
            try:
                probs = data_loader.load_precomputed_probabilities(N, figure_type='fig3')
            except Exception as e2:
                print(f"ERROR: Failed to load probabilities for N={N}")
                continue

        if probs is None:
            continue

        p_gg = probs[:, 0]
        p_eg = probs[:, 1]

        # 1. Enhanced Survival Probability Analysis

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Histogram with multiple distribution fits (Survival)
            n_bins = 50
            ax1.hist(p_gg, bins=n_bins, density=True, alpha=0.7,
                     color='lightblue', edgecolor='blue', label='Data')

            # Fits for Survival
            stat_analyzer = StatisticalAnalysis()
            x = np.linspace(np.min(p_gg), np.max(p_gg), 1000)  # Use actual data range for x
            d_min, d_max = np.min(p_gg), np.max(p_gg)  # Actual min/max
            scaled_p_gg = (p_gg - d_min) / (d_max - d_min)  # Scale data to [0,1]
            arcsine_params = stat_analyzer.fit_arcsine(scaled_p_gg)  # Fit on scaled [0,1]
            a, b = arcsine_params[:2]  # Extract a, b (ignore loc/scale if present)
            scaled_x = (x - d_min) / (d_max - d_min)  # Scale plot x to [0,1]
            arcsine_pdf = beta.pdf(scaled_x, a, b) / (d_max - d_min)  # Adjust PDF for original scale
            arcsine_eq = r'$f(x) = \frac{1}{\pi} \, \frac{1}{\sqrt{1 - x^2}}, \quad x \in [-1, 1]$'
            ax1.plot(x, arcsine_pdf, 'r-', label=f'Arcsine Fit {arcsine_eq}')
            mu, sigma = stat_analyzer.fit_normal(p_gg)
            ax1.plot(x, norm.pdf(x, mu, sigma), 'b--', label=r'Normal Fit $e^{-(x-\mu)^2/(2\sigma^2)}/\sqrt{2\pi\sigma^2}$')
            uniform_height = 1.0 / (x.ptp())            # ptp() = max-min
            ax1.plot(x, np.full_like(x, uniform_height), 'g-.', label=r'Uniform Fit \frac{1}{b-a}')

            expected_p_gg = 0.625 if N == 2 else 1.0/N
            ax1.axvline(expected_p_gg, color='orange', linestyle='-.', linewidth=2,
                        label=f'Expected Mean ({expected_p_gg:.3f})', alpha=0.8)
            ax1.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival Probability)', fontsize=12)
            ax1.set_ylabel('Probability Density', fontsize=12)
            ax1.set_title(f'Survival Probability Distribution (N={N})', fontsize=14)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Add statistical tests
            fit_stats = {
                'Arcsine Fit a' : f'{a:4f}',
                'Arcsine Fit b' : f'{b:4f}',
                'Normal Fit' : rf'\mu={mu:.4f}, \sigma={sigma:.4f}',
                'Uniform Fit' :rf'a={np.min(x)}, b={np.max(x)}'
            }
            add_statistics_textbox(ax1, fit_stats , 'lower right')

            # Add statistical tests
            ks_uniform = kstest(p_gg, 'uniform')
            survival_stats = {
                'Mean': f'{np.mean(p_gg):.4f}',
                'Expected': f'{expected_p_gg:.4f}',
                'Std Dev': f'{np.std(p_gg):.4f}',
                'KS vs Uniform': f'{ks_uniform.pvalue:.4f}'
            }
            add_statistics_textbox(ax1, survival_stats, 'upper right')

            # QQ plot for survival probability
            from scipy.stats import probplot
            probplot(p_gg, dist='uniform', plot=ax2)
            ax2.set_title(f'Q-Q Plot: Survival Prob vs Uniform (N={N})', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Conversion probability analysis with fits
            ax3.hist(p_eg, bins=n_bins, density=True, alpha=0.7,
                     color='lightgreen', edgecolor='green', label='Data')

            # Fits for Conversion
            # After ax3.hist(...):
            # Repeat identical structure as above, but for p_eg:
            x = np.linspace(np.min(p_eg), np.max(p_eg), 1000)
            d_min, d_max = np.min(p_eg), np.max(p_eg)
            scaled_p_eg = (p_eg - d_min) / (d_max - d_min)
            arcsine_params = stat_analyzer.fit_arcsine(scaled_p_eg)
            a, b = arcsine_params[:2]
            scaled_x = (x - d_min) / (d_max - d_min)
            arcsine_pdf = beta.pdf(scaled_x, a, b) / (d_max - d_min)
            arcsine_eq = rf'$\frac{{\Gamma({a:.2f} + {b:.2f})}}{{\Gamma({a:.2f}) \Gamma({b:.2f})}} (x - {d_min:.2f})^{{{a-1:.2f}}} ({d_max:.2f} - x)^{{{b-1:.2f}}} / ({d_max - d_min})^{{{a+b-1:.2f}}}$'
            ax3.plot(x, arcsine_pdf, 'r-', label=f'Arcsine Fit {arcsine_eq}')

            mu, sigma = stat_analyzer.fit_normal(p_eg)
            ax3.plot(x, norm.pdf(x, mu, sigma), 'b--', label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
            ax3.plot(x, np.full_like(x, 1.0), 'g-.', label='Uniform Fit')

            expected_p_eg = 0.25 if N == 2 else 0.1
            ax3.axvline(expected_p_eg, color='orange', linestyle='-.', linewidth=2,
                        label=f'Expected Mean ({expected_p_eg:.3f})', alpha=0.8)
            ax3.set_xlabel(r'$P_{e \to \gamma}$ (Conversion Probability)', fontsize=12)
            ax3.set_ylabel('Probability Density', fontsize=12)
            ax3.set_title(f'Conversion Probability Distribution (N={N})', fontsize=14)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Detailed stats for each fit (Conversion)
            arcsine_params = stat_analyzer.fit_arcsine(p_eg)
            ks_arcsine = kstest(p_eg, lambda x: beta.cdf(x, arcsine_params[0], arcsine_params[1]))
            normal_mu, normal_sigma = stat_analyzer.fit_normal(p_eg)
            ks_normal = kstest(p_eg, 'norm', args=(normal_mu, normal_sigma))
            ks_uniform = kstest(p_eg, 'uniform')

            arcsine_stats = {
                'Arcsine a': f'{arcsine_params[0]:.3f}',  # Correctly reference first element
                'Arcsine b': f'{arcsine_params[1]:.3f}',  # Correctly reference second element
                'Arcsine KS Stat': f'{ks_arcsine.statistic:.3f}',
                'Arcsine KS P-val': f'{ks_arcsine.pvalue:.3e}'
            }
            normal_stats = {
                'Normal μ': f'{normal_mu:.3f}',
                'Normal σ': f'{normal_sigma:.3f}',
                'Normal KS Stat': f'{ks_normal.statistic:.3f}',
                'Normal KS P-val': f'{ks_normal.pvalue:.3e}'
            }
            uniform_stats = {
                'Uniform KS Stat': f'{ks_uniform.statistic:.3f}',
                'Uniform KS P-val': f'{ks_uniform.pvalue:.3e}'
            }
            conversion_stats = {
                'Mean': f'{np.mean(p_eg):.4f}',
                'Expected': f'{expected_p_eg:.4f}',
                'Std Dev': f'{np.std(p_eg):.4f}',
                'Max Value': f'{np.max(p_eg):.4f}'
            }

            # Add multiple textboxes to avoid overlap
            add_statistics_textbox(ax3, arcsine_stats, 'upper right')
            add_statistics_textbox(ax3, normal_stats, 'upper left')
            add_statistics_textbox(ax3, uniform_stats, 'lower right')
            add_statistics_textbox(ax3, conversion_stats, 'lower left')


            # Scatter plot: Conversion vs Survival (Missing from original requirements!)
            ax4.scatter(p_gg, p_eg, alpha=0.6, s=5, c='purple')
            ax4.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival)', fontsize=12)
            ax4.set_ylabel(r'$P_{e \to \gamma}$ (Conversion)', fontsize=12)
            ax4.set_title(f'Conversion vs Survival Probability (N={N})', fontsize=14)
            ax4.grid(True, alpha=0.3)

            # Add correlation analysis
            correlation = np.corrcoef(p_gg, p_eg)[0, 1]
            corr_stats = {
                'Correlation': f'{correlation:.4f}',
                'Data Points': len(p_gg),
                'Independent?': 'Yes' if abs(correlation) < 0.1 else 'No'
            }
            add_statistics_textbox(ax4, corr_stats, 'upper right')

            plt.tight_layout()
            output_file = data_loader.PLOTS_DIR / f'probability_distributions_N{N}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_summary.append(["Probability Distributions", "Saved", output_file.name])
        except Exception as e:
            plot_summary.append(["Probability Distributions", "Failed", str(e)])
            plt.close()

        # 2. Matrix Elements Distribution (if available)

        try:
            matrix_data = data_loader.load_matrix_distributions(N)
            matrix_elements = matrix_data['matrix_elements']
            det_errors = matrix_data['det_errors']
            orth_errors = matrix_data['orth_errors']

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Matrix elements histogram with fits
            # Matrix elements histogram with fits
            ax1.hist(matrix_elements, bins=50, density=True, alpha=0.7,
                     color='lightcoral', edgecolor='red', label='Data')

            # Fits over full [-1,1] domain
            stat_analyzer = StatisticalAnalysis()
            x_fit = np.linspace(-1, 1, 1000)  # Full domain
            arcsine_params = stat_analyzer.fit_arcsine(matrix_elements)
            # Scale x_fit to [0,1] for PDF, then plot on original domain
            x_scaled = (x_fit + 1) / 2
            arcsine_pdf = beta.pdf(x_scaled, *arcsine_params) / 2  # Adjust PDF for scaling
            ax1.plot(x_fit, arcsine_pdf, 'r-', label=r'Arcsine Fit $\sqrt{1 - x^2}/\pi$')  # Equation in legend (no params)

            # Normal fit (already on [-1,1])
            mu, sigma = stat_analyzer.fit_normal(matrix_elements)
            ax1.plot(x_fit, norm.pdf(x_fit, mu, sigma), 'b--', label=r'Normal Fit $e^{-(x-\mu)^2/(2\sigma^2)}/\sqrt{2\pi\sigma^2}$')

            # Uniform fit
            uniform_min, uniform_max = -1, 1
            ax1.plot(x_fit, np.full_like(x_fit, 1/(uniform_max - uniform_min)), 'g-.', label='Uniform Fit 1/(b-a)')

            ax1.set_xlabel('Matrix Element Value', fontsize=12)
            ax1.set_ylabel('Probability Density', fontsize=12)
            ax1.set_title(f'SO(N) Matrix Elements Distribution (N={N})', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Stats box with parameters in LaTeX (no equation here)
            matrix_stats = {
                r'Arcsine $a$': f'{arcsine_params[0]:.4f}',  # LaTeX
                r'Arcsine $b$': f'{arcsine_params[1]:.4f}',  # LaTeX
                r'Normal $\mu$': f'{mu:.4f}',                 # LaTeX
                r'Normal $\sigma$': f'{sigma:.4f}',           # LaTeX
                r'Uniform Min': f'{uniform_min:.1f}',
                r'Uniform Max': f'{uniform_max:.1f}'
            }
            add_statistics_textbox(ax1, matrix_stats, 'upper right')



            # Determinant errors
            ax2.hist(det_errors, bins=30, density=True, alpha=0.7,
                    color='lightyellow', edgecolor='orange')
            stat_analyzer = StatisticalAnalysis()
            x_fit = np.linspace(np.min(matrix_elements), np.max(matrix_elements), 1000)
            # Arcsine (beta(0.5,0.5))
            arcsine_params = stat_analyzer.fit_arcsine(matrix_elements)
            ax2.plot(x_fit, beta.pdf(x_fit, *arcsine_params), 'g--', linewidth=2, label='Arcsine Fit (Beta(0.5,0.5))')

            # Normal
            mu, sigma = stat_analyzer.fit_normal(matrix_elements)
            ax2.plot(x_fit, norm.pdf(x_fit, mu, sigma), 'b-.', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')

            # Uniform
            uniform_min, uniform_max = np.min(matrix_elements), np.max(matrix_elements)
            ax2.plot(x_fit, np.full_like(x_fit, 1/(uniform_max - uniform_min)), 'r:', linewidth=2, label='Uniform Fit')

            ax2.legend(fontsize=10)  # Add/update legend to include fits
            ax2.set_xlabel('Determinant Error', fontsize=12)
            ax2.set_ylabel('Probability Density', fontsize=12)
            ax2.set_title(f'Determinant Accuracy (N={N})', fontsize=14)
            ax2.grid(True, alpha=0.3)

            det_stats = {
                'Mean Error': f'{np.mean(det_errors):.2e}',
                'Max Error': f'{np.max(det_errors):.2e}',
                'RMS Error': f'{np.sqrt(np.mean(det_errors**2)):.2e}'
            }
            add_statistics_textbox(ax2, det_stats, 'upper right')

            # Orthogonality errors
            ax3.hist(orth_errors, bins=30, density=True, alpha=0.7,
                    color='lightgreen', edgecolor='green')
            ax3.set_xlabel('Orthogonality Error', fontsize=12)
            ax3.set_ylabel('Probability Density', fontsize=12)
            ax3.set_title(f'Orthogonality Accuracy (N={N})', fontsize=14)
            ax3.grid(True, alpha=0.3)

            orth_stats = {
                'Mean Error': f'{np.mean(orth_errors):.2e}',
                'Max Error': f'{np.max(orth_errors):.2e}',
                'RMS Error': f'{np.sqrt(np.mean(orth_errors**2)):.2e}'
            }
            add_statistics_textbox(ax3, orth_stats, 'upper right')

            # Combined error analysis
            ax4.scatter(det_errors, orth_errors, alpha=0.6, s=2, c='purple')
            ax4.set_xlabel('Determinant Error', fontsize=12)
            ax4.set_ylabel('Orthogonality Error', fontsize=12)
            ax4.set_title(f'Matrix Quality Analysis (N={N})', fontsize=14)
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)

            # Add tolerance lines
            tolerance = CONSTANTS.get('MATRIX_TOLERANCE', 1e-12)
            ax4.axhline(tolerance, color='red', linestyle='--', alpha=0.8,
                       label=f'Tolerance ({tolerance:.0e})')
            ax4.axvline(tolerance, color='red', linestyle='--', alpha=0.8)
            ax4.legend(fontsize=10)

            plt.tight_layout()
            output_file = data_loader.PLOTS_DIR / f'matrix_distributions_N{N}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_summary.append(["Matrix Distributions", "Saved", output_file.name])
        except Exception as e:
            plot_summary.append(["Matrix Distributions", "Skipped", "Data not available"])
            plt.close()

        print(f"\nDiagnostic Distributions Summary (N={N}):")
        print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
        save_stats_to_file(plot_summary, ["Plot", "Status", "Output File"],
                          f"Diagnostic Distributions Summary (N={N})")


def plot_conversion_histograms(N_values, data_loader):
    """Generate separate conversion probability histograms with fits."""
    for N in N_values:
        try:
            print(f"DEBUG: Generating conversion histogram for N={N}...")
            probs = data_loader.load_precomputed_probabilities(N, figure_type='fig2')
            p_eg = probs[:, 1]

            fig, ax = plt.subplots(figsize=(10, 8))
            bins = 50
            ax.hist(p_eg, bins=bins, density=True, alpha=0.7, color='lightgreen', edgecolor='green', label='Observed Data')

            # Fits
            stat_analyzer = StatisticalAnalysis()
            arcsine_params = stat_analyzer.fit_arcsine(p_eg)
            print('arcsine_params : ', arcsine_params)
            a, b, loc, scale = arcsine_params  # guaranteed 4 elements from fit

            normal_mu, normal_sigma = stat_analyzer.fit_normal(p_eg)
            x = np.linspace(np.min(p_eg), np.max(p_eg), 1000)
            arcsine_eq = r'$\frac{\Gamma(a + b)}{\Gamma(a) \Gamma(b)} x^{a-1} (1 - x)^{b-1}$'
            ax.plot(x, beta.pdf(x, a, b, loc, scale), 'r-', label=f'Arcsine Fit {arcsine_eq}')
            normal_eq = r'$\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$'
            ax.plot(x, norm.pdf(x, normal_mu, normal_sigma), 'b--', label=f'Normal Fit {normal_eq}')
            uniform_eq = r'$1/(b - a)$'
            uniform_height = 1.0 / (x.ptp())
            ax.plot(x, np.full_like(x, uniform_height), 'g-.',label=f'Uniform Fit $1/({x.ptp():.2f})$')


            # Legend
            ax.legend(loc='upper left', fontsize=10)
            ax.set_xlabel(r'$P_{e \to \gamma}$ (Conversion Probability)', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(f'Conversion Probability Histogram (N={N})', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Separate fit table (positioned to not overlap legend)
            fit_stats = {
                'Arcsine Params': f'a={arcsine_params[0]:.3f}, b={arcsine_params[1]:.3f}',
                'Arcsine KS Stat': f'{kstest(p_eg, lambda x: beta.cdf(x, a, b, loc, scale)).statistic:.3f}',
                'Arcsine KS P-val': f'{kstest(p_eg, lambda x: beta.cdf(x, a, b, loc, scale)).pvalue:.3e}',
                'Normal μ': f'{normal_mu:.3f}',
                'Normal σ': f'{normal_sigma:.3f}',
                'Normal KS Stat': f'{kstest(p_eg, "norm", args=(normal_mu, normal_sigma)).statistic:.3f}',
                'Normal KS P-val': f'{kstest(p_eg, "norm", args=(normal_mu, normal_sigma)).pvalue:.3e}',
                'Uniform KS Stat': f'{kstest(p_eg, "uniform").statistic:.3f}',
                'Uniform KS P-val': f'{kstest(p_eg, "uniform").pvalue:.3e}'
            }
            add_statistics_textbox(ax, fit_stats, 'upper right')  # Existing position


            output_file = data_loader.PLOTS_DIR / f'conversion_hist_N{N}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"ERROR: Conversion histogram failed for N={N}: {str(e)}")
            plt.close()

def plot_convergence_and_diagnostics(data_loader):
    """Generate enhanced convergence analysis plots."""
    stat_analyzer = StatisticalAnalysis()
    plot_summary = []

    # Multi-N Convergence Analysis
    try:
        conv_data = data_loader.load_convergence_analysis()
        N_values = np.unique(conv_data['N'])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        # Convergence of survival probability
        for i, N in enumerate(N_values):
            N_data = conv_data['raw_data'][conv_data['N'] == N]
            n_real = N_data[:, 1]
            P_gg_mean = N_data[:, 2]
            P_gg_std = N_data[:, 3]

            expected_p_gamma = 0.625 if N == 2 else 1.0 / N
            color = colors[i % len(colors)]

            ax1.errorbar(n_real, P_gg_mean, yerr=P_gg_std,
                        label=f'N={int(N)} (exp: {expected_p_gamma:.3f})',
                        color=color, marker='o', linewidth=2, markersize=6)

            # Add expected value line
            ax1.axhline(y=expected_p_gamma, color=color, linestyle='--', alpha=0.5, label=f'Expected {expected_p_gamma:.3f}')

        ax1.set_xlabel('Number of Realizations', fontsize=12)
        ax1.set_ylabel(r'$\langle P_{\gamma \to \gamma} \rangle$', fontsize=12)
        ax1.set_title('Convergence of Survival Probability', fontsize=14)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Convergence of conversion probability
        for i, N in enumerate(N_values):
            N_data = conv_data['raw_data'][conv_data['N'] == N]
            n_real = N_data[:, 1]
            P_eg_mean = N_data[:, 4]
            P_eg_std = N_data[:, 5]

            expected_p_eg = 0.25 if N == 2 else 0.1
            color = colors[i % len(colors)]

            ax2.errorbar(n_real, P_eg_mean, yerr=P_eg_std,
                        label=f'N={int(N)} (exp: {expected_p_eg:.3f})',
                        color=color, marker='s', linewidth=2, markersize=6)

            ax2.axhline(y=expected_p_eg, color=color, linestyle='--', alpha=0.5, label=f'Expected {expected_p_eg:.3f}')

        ax2.set_xlabel('Number of Realizations', fontsize=12)
        ax2.set_ylabel(r'$\langle P_{e \to \gamma} \rangle$', fontsize=12)
        ax2.set_title('Convergence of Conversion Probability', fontsize=14)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # Relative error convergence
        for i, N in enumerate(N_values):
            N_data = conv_data['raw_data'][conv_data['N'] == N]
            n_real = N_data[:, 1]
            P_gg_mean = N_data[:, 2]

            expected_p_gamma = 0.625 if N == 2 else 1.0 / N
            relative_error = np.abs(P_gg_mean - expected_p_gamma) / expected_p_gamma
            color = colors[i % len(colors)]

            ax3.plot(n_real, relative_error, color=color, marker='o',
                    linewidth=2, markersize=6, label=f'N={int(N)}')

        ax3.set_xlabel('Number of Realizations', fontsize=12)
        ax3.set_ylabel('Relative Error in Mean', fontsize=12)
        ax3.set_title('Convergence of Relative Error', fontsize=14)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)

        # Add convergence criteria line
        ax3.axhline(0.01, color='red', linestyle='--', alpha=0.8,
                   label='1% Error Threshold')

        # Standard deviation convergence
        for i, N in enumerate(N_values):
            N_data = conv_data['raw_data'][conv_data['N'] == N]
            n_real = N_data[:, 1]
            P_gg_std = N_data[:, 3]

            # Theoretical standard error: std/sqrt(n)
            theoretical_std_error = P_gg_std[0] / np.sqrt(n_real / n_real[0])
            color = colors[i % len(colors)]

            ax4.plot(n_real, P_gg_std, color=color, marker='o',
                    linewidth=2, markersize=6, label=f'N={int(N)} (Observed)')
            ax4.plot(n_real, theoretical_std_error, color=color, linestyle='--',
                    alpha=0.7, label=f'N={int(N)} (Theoretical 1/√n)')

        ax4.set_xlabel('Number of Realizations', fontsize=12)
        ax4.set_ylabel('Standard Deviation', fontsize=12)
        ax4.set_title('Standard Error Convergence', fontsize=14)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / 'convergence_analysis_comprehensive.png'
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Comprehensive Convergence", "Saved", output_file.name])

    except Exception as e:
        plot_summary.append(["Comprehensive Convergence", "Failed", str(e)])
        plt.close()

    print("\nConvergence Analysis Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
    save_stats_to_file(plot_summary, ["Plot", "Status", "Output File"],
                      "Convergence Analysis Summary")

def plot_flux_comparisons(data_loader):
    """Enhanced flux comparison plots with comprehensive analysis."""
    stat_analyzer = StatisticalAnalysis()
    plot_summary = []

    print("\n" + "="*60)
    print("ENHANCED FLUX COMPARISON PLOTTING")
    print("="*60)

    # 1. Enhanced 2D Flux Plots
    try:
        # Load flux data
        flux_gg = data_loader.load_flux_vs_g_gamma()
        flux_ge = data_loader.load_flux_vs_g_e()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Flux vs g_gamma
        x_gg = flux_gg['log_g_gamma']
        ax1.loglog(10**x_gg, flux_gg['phi_p'], 'b-', linewidth=2, label='Primakoff')
        ax1.loglog(10**x_gg, flux_gg['phi_b'], 'g-', linewidth=2, label='Bremsstrahlung')
        ax1.loglog(10**x_gg, flux_gg['phi_c'], 'm-', linewidth=2, label='Compton')

        # Fit power laws and add them
        try:
            slope_p, int_p, r2_p = stat_analyzer.fit_power_law(10**x_gg, flux_gg['phi_p'])
            if r2_p > 0.1:
                fit_p = 10**(slope_p * x_gg + int_p)
                ax1.plot(10**x_gg, fit_p, 'b--', alpha=0.7,
                         label=f'Primakoff: slope={slope_p:.2f}')
        except:
            pass

        ax1.set_xlabel(r'$g_\gamma$ (GeV$^{-1}$)', fontsize=12)
        ax1.set_ylabel(r'Flux (m$^{-2}$ s$^{-1}$)', fontsize=12)
        ax1.set_title('Solar Axion Fluxes vs. Photon Coupling', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Flux vs g_e
        x_ge = flux_ge['log_g_e']
        ax2.loglog(10**x_ge, flux_ge['phi_p'] * np.ones_like(x_ge), 'b-',
                   linewidth=2, label='Primakoff (constant)')
        ax2.loglog(10**x_ge, flux_ge['phi_b'], 'g-', linewidth=2, label='Bremsstrahlung')
        ax2.loglog(10**x_ge, flux_ge['phi_c'], 'm-', linewidth=2, label='Compton')

        ax2.set_xlabel(r'$g_e$ (GeV$^{-1}$)', fontsize=12)
        ax2.set_ylabel(r'Flux (m$^{-2}$ s$^{-1}$)', fontsize=12)
        ax2.set_title('Solar Axion Fluxes vs. Electron Coupling', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Relative contributions (vs g_gamma)
        total_flux_gg = flux_gg['phi_p'] + flux_gg['phi_b'] + flux_gg['phi_c']
        valid_mask = total_flux_gg > 1e-50
        x_gg_valid = x_gg[valid_mask]
        total_flux_valid = total_flux_gg[valid_mask]
        primakoff_frac = flux_gg['phi_p'][valid_mask] / total_flux_valid
        bremsstrahlung_frac = flux_gg['phi_b'][valid_mask] / total_flux_valid
        compton_frac = flux_gg['phi_c'][valid_mask] / total_flux_valid

        ax3.semilogx(10**x_gg_valid, primakoff_frac, 'b-', linewidth=3,
                     label='Primakoff Fraction', alpha=0.9)
        ax3.semilogx(10**x_gg_valid, bremsstrahlung_frac, 'g-', linewidth=3,
                     label='Bremsstrahlung Fraction', alpha=0.9)
        ax3.semilogx(10**x_gg_valid, compton_frac, 'm-', linewidth=3,
                     label='Compton Fraction', alpha=0.9)

        ax3.set_xlabel(r'$g_\gamma$ (GeV$^{-1}$)', fontsize=12)
        ax3.set_ylabel('Relative Contribution', fontsize=12)
        ax3.set_title('Relative Flux Contributions vs. Photon Coupling', fontsize=14)
        ax3.legend(fontsize=10, loc='center right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        ax3.set_xlim(10**np.min(x_gg_valid), 10**np.max(x_gg_valid))

        # Relative contributions (vs g_e)
        total_flux_ge = flux_ge['phi_p'] + flux_ge['phi_b'] + flux_ge['phi_c']
        valid_mask_ge = total_flux_ge > 1e-50
        x_ge_valid = x_ge[valid_mask_ge]
        total_flux_ge_valid = total_flux_ge[valid_mask_ge]
        primakoff_frac_ge = flux_ge['phi_p'][valid_mask_ge] / total_flux_ge_valid
        bremsstrahlung_frac_ge = flux_ge['phi_b'][valid_mask_ge] / total_flux_ge_valid
        compton_frac_ge = flux_ge['phi_c'][valid_mask_ge] / total_flux_ge_valid

        ax4.semilogx(10**x_ge_valid, primakoff_frac_ge, 'b-', linewidth=3,
                     label='Primakoff Fraction', alpha=0.9)
        ax4.semilogx(10**x_ge_valid, bremsstrahlung_frac_ge, 'g-', linewidth=3,
                     label='Bremsstrahlung Fraction', alpha=0.9)
        ax4.semilogx(10**x_ge_valid, compton_frac_ge, 'm-', linewidth=3,
                     label='Compton Fraction', alpha=0.9)

        ax4.set_xlabel(r'$g_e$ (GeV$^{-1}$)', fontsize=12)
        ax4.set_ylabel('Relative Contribution', fontsize=12)
        ax4.set_title('Relative Flux Contributions vs. Electron Coupling', fontsize=14)
        ax4.legend(fontsize=10, loc='center right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
        ax4.set_xlim(10**np.min(x_ge_valid), 10**np.max(x_ge_valid))

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / 'flux_comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["Comprehensive Flux Analysis", "Saved", output_file.name])
    except Exception as e:
        plot_summary.append(["Comprehensive Flux Analysis", "Failed", str(e)])
        plt.close()

    # 2. Enhanced 3D Flux Analysis
    try:
        flux_3d = data_loader.load_flux_3d_grid()

        fig = plt.figure(figsize=(18,12))
        ax = fig.add_subplot(221, projection='3d')
        # 3D surface plot for Primakoff
        # 3D comparison plot of all flux types

        X, Y = np.meshgrid(flux_3d['log_g_e'], flux_3d['log_g_gamma'], indexing='ij')
        if not isinstance(flux_3d['phi_p'], np.ndarray):
            raise TypeError("flux_3d['phi_p'] must be a NumPy array")

        # Handle potential log(0) errors by clipping very small values
        Z_p = np.log10(np.clip(flux_3d['phi_p'], 1e-100, None))
        Z_b = np.log10(np.clip(flux_3d['phi_b'], 1e-100, None))
        Z_c = np.log10(np.clip(flux_3d['phi_c'], 1e-100, None))

        # --- FIX APPLIED HERE ---
        # Plot all three surfaces on the same axis using specific SOLID COLORS
        ax.plot_surface(X, Y, Z_p, color='purple', alpha=0.6, rstride=5, cstride=5)
        ax.plot_surface(X, Y, Z_b, color='orange', alpha=0.6, rstride=5, cstride=5)
        ax.plot_surface(X, Y, Z_c, color='teal', alpha=0.6, rstride=5, cstride=5)

        # Create proxy artists with matching SOLID COLORS to build a working legend
        primakoff_proxy = mpatches.Patch(color='purple', label='Primakoff Flux (Φ_p)')
        brems_proxy = mpatches.Patch(color='orange', label='Bremsstrahlung Flux (Φ_b)')
        compton_proxy = mpatches.Patch(color='teal', label='Compton Flux (Φ_c)')
        # ------------------------

        ax.legend(handles=[primakoff_proxy, brems_proxy, compton_proxy], fontsize=10, loc='upper right')

        # Set labels and title
        ax.set_xlabel(r'$\log_{10}(g_e)$', fontsize=12, labelpad=10)
        ax.set_ylabel(r'$\log_{10}(g_\gamma)$', fontsize=12, labelpad=10)
        ax.set_zlabel(r'$\log_{10}(\phi)$', fontsize=12, labelpad=10)
        ax.set_title('3D Comparison of Solar ALP Flux Components', fontsize=16)

        # --- HERE IS HOW TO CHANGE THE TICK LABEL SIZE ---
        ax.tick_params(axis='x', labelsize=6) # Increased size for better readability
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='z', labelsize=6)
        # ---------------------------------------------------

        ax.view_init(elev=30, azim=45)



        # Enhanced contour plots with CAST bounds
        ax2 = fig.add_subplot(222)
        levels = np.logspace(np.log10(np.min(flux_3d['phi_p'])),
                             np.log10(np.max(flux_3d['phi_p'])), 15)
        contour_p = ax2.contourf(X, Y, flux_3d['phi_p'].T, levels=levels,
                                 cmap='viridis', norm=LogNorm())
        # Add CAST bound line
        g_e_vals = 10**flux_3d['log_g_e']
        cast_line = np.log10([g_gamma_n1(ge) for ge in g_e_vals])
        ax2.plot(flux_3d['log_g_e'], cast_line, 'r--', linewidth=3, label='CAST Constraint')
        ax2.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=12)
        ax2.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=12)
        ax2.set_title('Primakoff Flux with CAST Bound', fontsize=14)
        ax2.legend(fontsize=10)
        plt.colorbar(contour_p, ax=ax2, label='Flux (m$^{-2}$ s$^{-1}$)')

        # Dominant production mechanism map
        ax3 = fig.add_subplot(223)
        phi_total = flux_3d['phi_p'] + flux_3d['phi_b'] + flux_3d['phi_c']
        dominant = np.zeros_like(phi_total)
        # 0 = Primakoff, 1 = Bremsstrahlung, 2 = Compton
        max_flux = np.maximum(np.maximum(flux_3d['phi_p'], flux_3d['phi_b']), flux_3d['phi_c'])
        dominant[flux_3d['phi_p'] == max_flux] = 0
        dominant[flux_3d['phi_b'] == max_flux] = 1
        dominant[flux_3d['phi_c'] == max_flux] = 2
        im = ax3.contourf(X, Y, dominant.T, levels=[0, 0.5, 1.5, 2.5],
                          colors=['blue', 'green', 'magenta'], alpha=0.7)
        ax3.plot(flux_3d['log_g_e'], cast_line, 'r--', linewidth=3, label='CAST Constraint')
        ax3.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=12)
        ax3.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=12)
        ax3.set_title('Dominant Production Mechanism', fontsize=14)
        # Custom legend for dominant mechanism
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Primakoff'),
                           Patch(facecolor='green', label='Bremsstrahlung'),
                           Patch(facecolor='magenta', label='Compton')]
        ax3.legend(handles=legend_elements, fontsize=10)

        # Total flux contours
        ax4 = fig.add_subplot(224)
        total_levels = np.logspace(np.log10(np.min(phi_total)),
                                   np.log10(np.max(phi_total)), 15)
        contour_total = ax4.contourf(X, Y, phi_total.T, levels=total_levels,
                                     cmap='plasma', norm=LogNorm())
        ax4.plot(flux_3d['log_g_e'], cast_line, 'k--', linewidth=3, label='CAST Constraint')
        ax4.set_xlabel(r'$\log_{10}(g_e / \mathrm{GeV}^{-1})$', fontsize=12)
        ax4.set_ylabel(r'$\log_{10}(g_\gamma / \mathrm{GeV}^{-1})$', fontsize=12)
        ax4.set_title('Total Solar Axion Flux', fontsize=14)
        ax4.legend(fontsize=10)
        plt.colorbar(contour_total, ax=ax4, label='Total Flux (m$^{-2}$ s$^{-1}$)')

        plt.tight_layout()
        output_file = data_loader.PLOTS_DIR / 'flux_3d_comprehensive.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_summary.append(["3D Flux Comprehensive", "Saved", output_file.name])
    except Exception as e:
        plot_summary.append(["3D Flux Comprehensive", "Failed", str(e)])
        plt.close()

    print("\nFlux Comparison Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
    save_stats_to_file(plot_summary, ["Plot", "Status", "Output File"], "Flux Comparison Summary")

    return plot_summary


def plot_cdf_and_statistical_tests(N_values, data_loader):
    """Generate enhanced CDF plots with comprehensive statistical testing."""
    plot_summary = []
    stat_analyzer = StatisticalAnalysis()

    for N in N_values:
        try:
            # Load CDF data
            cdf_data = data_loader.load_cdf_p_gg(N, figure_type='fig2')
            p_gg = cdf_data['P_gg']
            cdf = cdf_data['CDF']

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Empirical CDF with exponential fit
            ax1.plot(p_gg, cdf, 'b-', linewidth=2, label='Empirical CDF')

            # Fit exp(a x + b) to CDF data
            def exp_fit_func(x, a, b):
                return np.exp(a * x + b)

            # Clip y to avoid log(0) or invalid values
            cdf_clip = np.clip(cdf, 1e-10, np.inf)
            params, _ = curve_fit(exp_fit_func, p_gg, cdf_clip, p0=[1, 0], maxfev=5000)
            a_fit, b_fit = params

            x_theory = np.linspace(0, np.max(p_gg), 1000)
            exp_fit = exp_fit_func(x_theory, a_fit, b_fit)
            equation_str = r'$\exp(ax + b)$'  # Symbolic
            ax1.plot(x_theory, exp_fit, 'm--', linewidth=2, label=f'Exponential Fit {equation_str}')

            # Sigmoid fit
            def sigmoid_func(x, a, b, c, d):
                return a / (1 + np.exp(-b * (x - c))) + d

            y_clip = np.clip(cdf, 1e-4, 1-1e-4)
            sigmoid_params, _ = curve_fit(sigmoid_func, p_gg, y_clip, p0=[1, 1, np.mean(p_gg), 0], maxfev=5000)
            sigmoid_a, sigmoid_b, sigmoid_c, sigmoid_d = sigmoid_params
            sigmoid_fit = sigmoid_func(x_theory, sigmoid_a, sigmoid_b, sigmoid_c, sigmoid_d)
            sigmoid_eq = r'$\frac{a}{1 + e^{-b(x - c)}} + d$'
            # After computing sigmoid_params
            if np.all(np.isfinite(sigmoid_params)):
                ax1.plot(x_theory, sigmoid_fit, 'c--', linewidth=2, label=f'Sigmoid Fit {sigmoid_eq}')



            ax1.set_xlabel(r'$P_{\gamma \to \gamma}$', fontsize=12)
            ax1.set_ylabel('Cumulative Probability', fontsize=12)
            ax1.set_title(f'Cumulative Distribution Function (N={N})', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # KS test for exponential fit and stats table (LaTeX format)
            ks_exp = kstest(cdf, lambda x: exp_fit_func(x, a_fit, b_fit))
            cdf_stats = {
                'KS Stat (Exp)': f'{ks_exp.statistic:.4f}',
                'KS P-value (Exp)': f'{ks_exp.pvalue:.4f}',
                'Sample Size': len(p_gg),
                '$a$': f'{a_fit:.3f}',  # LaTeX
                '$b$': f'{b_fit:.3f}'   # LaTeX
            }
            add_statistics_textbox(ax1, cdf_stats, 'lower right')  # Position below legend

            sigmoid_stats = {
                '$a$': f'{sigmoid_a:.3f}',
                '$b$': f'{sigmoid_b:.3f}',
                '$c$': f'{sigmoid_c:.3f}',
                '$d$': f'{sigmoid_d:.3f}'
            }
            add_statistics_textbox(ax1, sigmoid_stats, 'upper right')



            # 2. P-P plot
            uniform_quantiles = np.linspace(0, 1, len(p_gg))
            sorted_p_gg = np.sort(p_gg)
            ax2.plot(uniform_quantiles, sorted_p_gg, 'bo', markersize=3, alpha=0.6)
            ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
            ax2.set_xlabel('Uniform Quantiles', fontsize=12)
            ax2.set_ylabel('Sample Quantiles', fontsize=12)
            ax2.set_title(f'P-P Plot vs Uniform Distribution (N={N})', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            # 3. KS test visualization
            empirical_cdf = np.arange(1, len(p_gg) + 1) / len(p_gg)
            ax3.plot(sorted_p_gg, empirical_cdf, 'b-', linewidth=2, label='Empirical CDF')
            ax3.plot(sorted_p_gg, sorted_p_gg, 'r--', linewidth=2, label='Uniform CDF')

            # Find maximum difference
            max_diff_idx = np.argmax(np.abs(empirical_cdf - sorted_p_gg))
            max_diff = np.abs(empirical_cdf[max_diff_idx] - sorted_p_gg[max_diff_idx])
            ax3.vlines(sorted_p_gg[max_diff_idx], sorted_p_gg[max_diff_idx],
                       empirical_cdf[max_diff_idx], colors='red', linewidth=3,
                       label=f'Max Diff = {max_diff:.3f}')

            ax3.set_xlabel(r'$P_{\gamma \to \gamma}$', fontsize=12)
            ax3.set_ylabel('Cumulative Probability', fontsize=12)
            ax3.set_title(f'KS Test Visualization (N={N})', fontsize=14)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

            # 4. Distribution comparison histogram
            probs = data_loader.load_precomputed_probabilities(N, figure_type='fig2')
            p_gg_hist = probs[:, 0]
            ax4.hist(p_gg_hist, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='blue', label='Data')

            # Overlays with symbolic equations
            x = np.linspace(np.min(p_gg_hist), np.max(p_gg_hist), 200)
            expected_mean = 0.625 if N == 2 else 1.0 / N
            uniform_height = 1.0 / (np.max(p_gg_hist) - np.min(p_gg_hist))     # use DATA domain
            ax4.plot(x, np.full_like(x, uniform_height), 'r--', linewidth=2, label='Uniform Fit $1/(b-a)$')
            ax4.plot(x, beta.pdf(x, 0.5, 0.5), 'g--', linewidth=2, label=r'Theoretical Arcsine $a=0.5, b=0.5$')

            # NEW: Robust Beta fit with scaling and fallback
            x_min, x_max = p_gg_hist.min(), p_gg_hist.max()
            if x_max > x_min:  # Avoid division by zero
                scaled = (p_gg_hist - x_min) / (x_max - x_min)
                pars0 = (0.5, 0.5, 0.0, 1.0)  # Initial guess: symmetric Arcsine
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')  # Suppress overflow warnings
                        a_fit, b_fit, loc_fit, scale_fit = beta.fit(scaled, *pars0, maxfev=20000)
                    # Plot back in original scale
                    scaled_x = (x - x_min) / (x_max - x_min)
                    beta_pdf = beta.pdf(scaled_x, a_fit, b_fit, loc_fit, scale_fit) / (x_max - x_min)
                    ax4.plot(x, beta_pdf, 'r-', linewidth=2, label=r'Actual beta Fit to Data $\frac{\Gamma(a + b)}{\Gamma(a) \Gamma(b)} x^{a-1} (1 - x)^{b-1}$')
                except RuntimeError:
                    # Fallback to analytical arcsine if fit fails
                    a_fit, b_fit = 0.5, 0.5
                    scaled_x = (x - x_min) / (x_max - x_min)
                    beta_pdf = beta.pdf(scaled_x, a_fit, b_fit) / (x_max - x_min)
                    ax4.plot(x, beta_pdf, 'r--', linewidth=2, label='Arcsine (fallback)')
            else:
                # If data is constant (degenerate case), skip Beta fit
                ax4.plot(x, np.zeros_like(x), 'r--', linewidth=2, label='Beta Fit Skipped (degenerate data)')

            mu, sigma = stat_analyzer.fit_normal(p_gg_hist)
            ax4.plot(x, norm.pdf(x, mu, sigma), 'b-.', linewidth=2, label='Normal Fit $e^{-(x-\\mu)^2/(2\\sigma^2)}/\\sqrt{2\\pi\\sigma^2}$')
            ax4.axvline(expected_mean, color='orange', linestyle='-.', linewidth=2, label=f'Expected Mean ({expected_mean:.3f})')
            ax4.set_xlabel(r'$P_{\gamma \to \gamma}$', fontsize=12)
            ax4.set_ylabel('Probability Density', fontsize=12)
            ax4.set_title(f'Distribution Comparison (N={N})', fontsize=14)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

            # Stats table below legend with LaTeX
            fit_stats = {
                'Beta Function Params': f'a={a_fit:.2f}, b={b_fit:.2f})',
                'Normal μ': f'{mu:.3f}',
                'Normal σ': f'{sigma:.3f}',
                'Uniform': rf'PDF=1 [{np.min(p_gg_hist)}, {np.max(p_gg_hist)}]',
            }
            add_statistics_textbox(ax4, fit_stats, 'lower right')

            plt.tight_layout()
            output_file = data_loader.PLOTS_DIR / f'cdf_statistical_analysis_N{N}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            plot_summary.append([f"CDF Analysis N={N}", "Saved", output_file.name])

        except Exception as e:
            plot_summary.append([f"CDF Analysis N={N}", "Failed", str(e)])
            plt.close()

    print("\nCDF and Statistical Tests Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))
    save_stats_to_file(plot_summary, ["Plot", "Status", "Output File"],
                       "CDF and Statistical Tests Summary")

def plot_realization_diagnostics(N_values, data_loader):
    """Generate realization diagnostic scatter plots (Pe→γ vs Pγ→γ)."""
    plot_summary = []
    for N in N_values:
        try:
            # Load probability data
            probs = data_loader.load_precomputed_probabilities(N, figure_type='fig2')
            p_gg = probs[:, 0]  # Survival probability
            p_eg = probs[:, 1]  # Conversion probability

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Main scatter plot: Conversion vs Survival
            ax1.scatter(p_gg, p_eg, alpha=0.6, s=4, c='purple', rasterized=True)
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
            hb = ax2.hexbin(p_gg, p_eg, gridsize=40, cmap='Blues', mincnt=1)
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
            add_statistics_textbox(ax3, survival_dist_stats, 'upper right')

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
            add_statistics_textbox(ax4, conversion_dist_stats, 'upper right')


            plt.tight_layout()
            output_file = data_loader.PLOTS_DIR / f'realization_diagnostics_N{N}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plot_summary.append(["Realization Diagnostics N="+str(N), "Saved", output_file.name])
            plt.close()


        except Exception as e:
            print(f"ERROR: Realization diagnostics failed for N={N}: {str(e)}")
            plt.close()

    print("\nRealization Diagnostics Summary:")
    print(tabulate(plot_summary, headers=["Plot", "Status", "Output File"], tablefmt="grid"))



def generate_master_summary_plot(data_loader):
    """Generate a master summary plot combining key results."""
    try:
        fig = plt.figure(figsize=(28, 22))
        fig.subplots_adjust(top=0.85, bottom=0.10, hspace=0.35, wspace=0.25)
        # then add a tight-layout friendly suptitle
        fig.suptitle('ALP Anarchy Analysis: Master Summary', fontsize=12, y=0.95)          # y sets title height

        # Load key data
        fig2_data_2 = data_loader.load_figure_2_data(2)
        fig2_data_30 = data_loader.load_figure_2_data(30)
        fig3_data = data_loader.load_figure_3_data()

        N_range = fig3_data['N_range']
        log_g_50 = fig3_data['log_g_50gamma']
        log_N = np.log10(N_range)

        # Perform linear fit (consistent with individual plot)
        slope, intercept, r_value, p_value, std_err = linregress(log_N, log_g_50)
        r_squared = r_value**2
        fit_line = slope * log_N + intercept

        # Theoretical line (consistent with individual plot)
        theoretical_slope = 0.25
        theoretical_intercept = -(23/np.log(10))
        theoretical_fit = theoretical_slope * log_N + theoretical_intercept

        # Figure 2 main heatmap for N=2
        ax1 = plt.subplot(3, 3, 1)
        W_2 = fig2_data_2['W'].T
        g_e_range_2 = fig2_data_2['g_e_range']
        g_gamma_range_2 = fig2_data_2['g_gamma_range']
        im1 = ax1.pcolormesh(g_e_range_2, g_gamma_range_2, W_2, cmap='viridis', vmin=0, vmax=1, shading='gouraud')
        cast_patch = mpatches.Patch(color='red', label='CAST Constraint')
        ax1.legend(handles=[cast_patch], fontsize=8)
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Fraction of Realisations')  # Added label
        g_n1_vals_2 = np.array([g_gamma_n1(10**ge) for ge in g_e_range_2])
        cast_line_y_2 = np.log10(g_n1_vals_2)
        ax1.plot(g_e_range_2, cast_line_y_2, 'r-', linewidth=2, alpha=0.8)
        ax1.set_xlabel(r'$\log_{10}(g_e)$', fontsize=10)
        ax1.set_ylabel(r'$\log_{10}(g_\gamma)$', fontsize=10)
        ax1.set_title('Figure 2: Main Heatmap (N=2)', fontsize=12)

        # Figure 2 main heatmap for N=30
        ax1_30 = plt.subplot(3, 3, 3)
        W_30 = fig2_data_30['W'].T
        g_e_range_30 = fig2_data_30['g_e_range']
        g_gamma_range_30 = fig2_data_30['g_gamma_range']
        im1_30 = ax1_30.pcolormesh(g_e_range_30, g_gamma_range_30, W_30, cmap='viridis', vmin=0, vmax=1, shading='gouraud')
        cast_patch_30 = mpatches.Patch(color='red', label='CAST Constraint')  # Added legend box
        ax1_30.legend(handles=[cast_patch_30], fontsize=8)
        cbar2 = plt.colorbar(im1_30, ax=ax1_30)
        cbar2.set_label('Fraction of Realisations')  # Added label
        g_n1_vals_30 = np.array([g_gamma_n1(10**ge) for ge in g_e_range_30])
        cast_line_y_30 = np.log10(g_n1_vals_30)
        ax1_30.plot(g_e_range_30, cast_line_y_30, 'r--', linewidth=2, alpha=0.8)
        ax1_30.set_xlabel(r'$\log_{10}(g_e)$', fontsize=10)
        ax1_30.set_ylabel(r'$\log_{10}(g_\gamma)$', fontsize=10)
        ax1_30.set_title('Figure 2: Main Heatmap (N=30)', fontsize=12)

        # Replaced Figure 3: Scaling Law (only calculated fit line, log-log)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(log_N, log_g_50, 'bo', markersize=6, label='Simulation Data')
        ax2.plot(log_N, fit_line, 'r-', linewidth=3, alpha=0.8,
                 label=f'Fit: $y = {slope:.4f} \\log_{{10}}(N) + {intercept:.4f}$')
        ax2.set_xlabel(r'$\log_{10}(N)$', fontsize=10)
        ax2.set_ylabel(r'$\log_{10}(g_{\gamma,50})$', fontsize=10)
        ax2.set_title('Figure 3: Scaling Law', fontsize=12)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Add stats box
        fit_stats = {
            'Fitted Slope': f'{slope:.4f}',
            'Fitted Intercept': f'{intercept:.4f}',
            '$R^2$': f'{r_squared:.4f}'
        }
        add_statistics_textbox(ax2, fit_stats, 'upper right')

        # Replaced Figure 3: Scaling Law (Theoretical Only) (only theoretical line, log-log)
        ax4 = plt.subplot(3, 3, 5)
        ax4.plot(log_N, log_g_50, 'bo', markersize=6, label='Simulation Data')
        ax4.plot(log_N, theoretical_fit, 'g--', linewidth=3, alpha=0.8,
                 label=f'Theory: $y = 0.25 \\log_{{10}}(N) + {theoretical_intercept:.4f}$')
        ax4.set_xlabel(r'$\log_{10}(N)$', fontsize=10)
        ax4.set_ylabel(r'$\log_{10}(g_{\gamma,50})$', fontsize=10)
        ax4.set_title('Figure 3: Scaling Law (Theoretical Only)', fontsize=12)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # New: log(g_50,gamma) vs N with calculated fit (linear N scale)
        ax5 = plt.subplot(3, 3, 4)
        ax5.plot(N_range, log_g_50, 'bo', markersize=6, label='Simulation Data')
        fit_line_lin = slope * np.log10(N_range) + intercept  # Recalculate for linear x
        ax5.plot(N_range, fit_line_lin, 'r-', linewidth=3, alpha=0.8,
                 label=f'Fit: $y = {slope:.4f} \\log_{{10}}(N) + {intercept:.4f}$')
        ax5.set_xlabel('N', fontsize=10)
        ax5.set_ylabel(r'$\log_{10}(g_{\gamma,50})$', fontsize=10)
        ax5.set_title('Figure 3: Scaling Law (Calculated Fit, Linear N)', fontsize=12)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # New: log(g_50,gamma) vs N with theoretical fit (linear N scale)
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(N_range, log_g_50, 'bo', markersize=6, label='Simulation Data')
        theoretical_fit_lin = theoretical_slope * np.log10(N_range) + theoretical_intercept
        ax6.plot(N_range, theoretical_fit_lin, 'g--', linewidth=3, alpha=0.8,
                 label=f'Theory: $y = 0.25 \\log_{{10}}(N) + {theoretical_intercept:.4f}$')
        ax6.set_xlabel('N', fontsize=10)
        ax6.set_ylabel(r'$\log_{10}(g_{\gamma,50})$', fontsize=10)
        ax6.set_title('Figure 3: Scaling Law (Theoretical Fit, Linear N)', fontsize=12)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rect reserves space for suptitle

        output_file = data_loader.PLOTS_DIR / 'master_summary_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return [["Master Summary", "Saved", "Summarised Results of Analysis"]]

    except Exception as e:
           plt.close()
           return [["Master Summary", "Failed", str(e)]]



def plot_combined_cdf(N_values, data_loader):
    """Generate combined CDF plot for various N in one canvas."""
    print("DEBUG: Generating combined CDF plot...")

    fig, ax = plt.subplots(figsize=(20, 16))  # Increased size to avoid overlap

    positions = ['upper left', 'upper right', 'lower right']  # Different positions for stats boxes
    colors = ['blue', 'green', 'red']  # Color code lines for each N

    for idx, N in enumerate(N_values):
        try:
            cdf_data = data_loader.load_cdf_p_gg(N)
            p_gg = cdf_data['P_gg']
            cdf = cdf_data['CDF']
            ax.plot(p_gg, cdf, color=colors[idx % len(colors)], label=f'N={N}', linewidth=2)

            # Add KS test stats for each N
            stat_analyzer = StatisticalAnalysis()
            ks_stat, ks_pval = stat_analyzer.kolmogorov_smirnov_test(p_gg, 'arcsine')
            stats = {
                f'KS Stat (N={N})': f'{ks_stat:.3f}',
                f'KS P-value (N={N})': f'{ks_pval:.3e}'
            }
            add_statistics_textbox(ax, stats, positions[idx % len(positions)])
        except Exception as e:
            print(f"ERROR: Failed to load CDF data for N={N}: {str(e)}")

    ax.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival Probability)', fontsize=14)
    ax.set_ylabel('Cumulative Distribution Function', fontsize=14)
    ax.set_title('Combined CDF Distributions vs Probabilities for Various N', fontsize=16)
    ax.legend(fontsize=12, loc='center', bbox_to_anchor=(1.0, 0.0))  # Shift legend to bottom-right corner
    ax.grid(True, alpha=0.3)

    output_file = data_loader.PLOTS_DIR / 'combined_cdf_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"DEBUG: Saved combined CDF plot to {output_file}")


def plot_figure3_cdf(data_loader):
    """
    Generates a CDF plot of Figure 3 precomputed survival probabilities for each N.

    - Loads precomputed probability files for each N from Figure 3 data.
    - Plots CDF of P_gg across all available N values.
    - Annotates each curve with its N label and shows Kolmogorov-Smirnov statistics.
    """

    # List of N values from figure 3 used in the simulation (typically N=2,10,30)
    N_values = list(range(2,31,1))

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap('viridis')


    for idx, N in enumerate(N_values):
        # Load probability data for Figure 3 (P_gg for this N)
        try:
            probs = data_loader.load_precomputed_probabilities(N, figure_type='fig3')
        except Exception:
            continue
        p_gg = np.sort(probs[:, 0])
        cdf = np.arange(1, len(p_gg) + 1) / len(p_gg)
        color_val = (N - min(N_values)) / (max(N_values) - min(N_values))
        ax.plot(p_gg, cdf, color=cmap(color_val), label=f'N={N}', linewidth=2)
        # KS test versus arcsine or uniform distribution



    ax.set_xlabel(r'$P_{\gamma \to \gamma}$ (Survival Probability)', fontsize=14)
    ax.set_ylabel('Cumulative Distribution Function', fontsize=14)
    ax.set_title('Figure 3: Survival Probability CDF for Multiple N', fontsize=16)
    # Add a color bar to serve as a legend for N
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(N_values), vmax=max(N_values)))
    sm.set_array([]) # You must set the array for the colorbar to work.
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Number of ALPs (N)', fontsize=14)
    ax.grid(True, alpha=0.3)
    output_file = data_loader.PLOTS_DIR / 'figure3_cdf_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
