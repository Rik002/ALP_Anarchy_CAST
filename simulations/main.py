import time
import os
from tqdm import tqdm
from src_python.data_loading import DataLoading
from src_python.plotting import generate_figure_2, generate_figure_3, plot_distributions, plot_diagnostic_distributions_fig2, plot_diagnostic_distributions_fig3, generate_convergence_plots, diagnose_figure_3
from tabulate import tabulate

# Debug: Confirm data_loading.py path
print(f"DEBUG: Using data_loading.py from: {os.path.abspath(DataLoading.__module__ + '.py')}")

def main():
    start_time = time.time()
    print("Starting ALP Anarchy CAST Analysis Pipeline")
    data_loader = DataLoading()
    summary = []

    try:
        N_values = [2, 10, 30]
        for N in tqdm(N_values, desc="Generating Figure 2"):
            t_start = time.time()
            generate_figure_2(N, data_loader)
            elapsed = time.time() - t_start
            summary.append([f"Figure 2 (N={N})", "Completed", f"{elapsed:.2f}"])

        t_start = time.time()
        generate_figure_3(data_loader)
        elapsed = time.time() - t_start
        summary.append(["Figure 3", "Completed", f"{elapsed:.2f}"])

        t_start = time.time()
        plot_distributions(N_values, data_loader)
        elapsed = time.time() - t_start
        summary.append(["Distributions", "Completed", f"{elapsed:.2f}"])

        t_start = time.time()
        plot_diagnostic_distributions_fig2(N_values, data_loader)
        elapsed = time.time() - t_start
        summary.append(["Diagnostic Fig2", "Completed", f"{elapsed:.2f}"])

        t_start = time.time()
        plot_diagnostic_distributions_fig3(data_loader)
        elapsed = time.time() - t_start
        summary.append(["Diagnostic Fig3", "Completed", f"{elapsed:.2f}"])

        t_start = time.time()
        generate_convergence_plots(data_loader)
        elapsed = time.time() - t_start
        summary.append(["Convergence Plots", "Completed", f"{elapsed:.2f}"])

        t_start = time.time()
        diagnose_figure_3(data_loader)
        elapsed = time.time() - t_start
        summary.append(["Figure 3 Diagnostics", "Completed", f"{elapsed:.2f}"])

        elapsed_total = time.time() - start_time
        print("\nAnalysis Summary:")
        print(tabulate(summary, headers=["Task", "Status", "Time (seconds)"], tablefmt="grid"))
        print(f"Total analysis time: {elapsed_total:.2f} seconds")
        print("Plots are stored in plots/ folder")
    except Exception as e:
        print(f"Error: {str(e)}")
        summary.append(["Pipeline", "Failed", "N/A"])
        print("\nAnalysis Summary:")
        print(tabulate(summary, headers=["Task", "Status", "Time (seconds)"], tablefmt="grid"))
        print("Plots are stored in plots/ folder")
        exit(1)

if __name__ == "__main__":
    main()
