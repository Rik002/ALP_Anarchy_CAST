#include "../include_cpp/simulations.h"
#include "../include_cpp/constants.h"
#include "../include_cpp/flux_calculations.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize terminal-safe output
    const char* lang = std::getenv("LANG");
    bool use_unicode = (lang && (strstr(lang, "UTF-8") || strstr(lang, "utf8")));

    std::cout << "======================================================================" << std::endl;
    std::cout << "ALP Anarchy CAST Simulation Pipeline - Enhanced Version" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "Comprehensive simulation with statistical validation and diagnostics" << std::endl;
    std::cout << "Generated: " << DataOutput::generate_timestamp() << std::endl;
    std::cout << "Unicode support: " << (use_unicode ? "YES" : "NO (ASCII mode)") << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<std::string> summary;

    try {
        // Load constants with enhanced error checking
        std::cout << "[INIT] Loading configuration from constants.dat..." << std::endl;
        if (!Constants::load_from_file("../config/constants.dat")) {
            std::cerr << "[ERROR] Failed to load constants.dat" << std::endl;
            return 1;
        }

        std::cout << "[SUCCESS] Constants loaded successfully." << std::endl;
        std::cout << "[CONFIG] Simulation Parameters:" << std::endl;
        std::cout << "         N_REALIZATIONS: " << Constants::N_REALIZATIONS << std::endl;
        std::cout << "         MAX_N: " << Constants::MAX_N << std::endl;
        std::cout << "         N_THREADS: " << Constants::N_THREADS << std::endl;
        std::cout << "         Output directory: " << Constants::DATA_DIR << std::endl;
        std::cout << "         G_E range: [" << Constants::G_E_LOG_MIN << ", " << Constants::G_E_LOG_MAX << "] (log10)" << std::endl;
        std::cout << "         G_GAMMA range: [" << Constants::G_GAMMA_LOG_MIN << ", " << Constants::G_GAMMA_LOG_MAX << "] (log10)" << std::endl;
        std::cout << "         Matrix tolerance: " << std::scientific << Constants::MATRIX_TOLERANCE << std::endl;
        std::cout << std::endl;

        // Display adaptive validation criteria
        std::cout << "[VALIDATION] Adaptive tolerance thresholds:" << std::endl;
        std::cout << "            <= 1000 realizations: " << Constants::PHYSICS_TOLERANCE_LOOSE * 100 << "%" << std::endl;
        std::cout << "            1000-9999 realizations: " << Constants::PHYSICS_TOLERANCE_MEDIUM * 100 << "%" << std::endl;
        std::cout << "            >= 10000 realizations: " << Constants::PHYSICS_TOLERANCE_STRICT * 100 << "%" << std::endl;
        std::cout << "            Current tolerance: " << Constants::get_validation_tolerance(Constants::N_REALIZATIONS) * 100 << "%" << std::endl;
        std::cout << std::endl;

        // Create output directory
        std::cout << "[INIT] Creating output directory: " << Constants::DATA_DIR << std::endl;
        if (!DataOutput::create_output_directory(Constants::DATA_DIR)) {
            std::cerr << "[ERROR] Failed to create output directory" << std::endl;
            return 1;
        }

        // Validate flux calculations
        std::cout << "[VALIDATION] Validating flux calculations..." << std::endl;
        if (!FluxCalculations::validate_flux_calculations()) {
            std::cerr << "[ERROR] Flux calculation validation failed" << std::endl;
            return 1;
        }
        std::cout << "[SUCCESS] Flux calculations validated successfully." << std::endl;
        std::cout << std::endl;

        // Generate Figure 2 data for different N values
        int N_values[] = {2,10, 30};
        for (int N : N_values) {
            std::cout << "[SIMULATION] Starting Figure 2 generation for N=" << N << std::endl;
            auto t_start = std::chrono::high_resolution_clock::now();

            // Expected physics results with ASCII-safe formatting
            double expected_p_gamma = (N == 2) ? 0.625 : 1.0/N;
            double expected_p_e_gamma = (N == 2) ? 0.25 : 0.1;

            std::cout << "[PHYSICS] Expected P_gg: " << std::scientific << std::setprecision(3) << expected_p_gamma << std::endl;
            std::cout << "[PHYSICS] Expected P_eg: " << std::scientific << std::setprecision(3) << expected_p_e_gamma << std::endl;

            Simulations::generate_figure_2(N);

            auto t_end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(t_end - t_start).count();
            summary.push_back("Figure 2 (N=" + std::to_string(N) + ") | Completed | " + DataOutput::format_duration(duration));

            std::cout << "[SUCCESS] Figure 2 (N=" << N << ") completed in " << DataOutput::format_duration(duration) << std::endl;
            std::cout << std::endl;
        }

        // Generate Figure 3 scaling analysis
        std::cout << "[SIMULATION] Starting Figure 3 scaling analysis" << std::endl;
        std::cout << "[PHYSICS] Expected scaling: g_gamma_50(N) ~ N^(1/4)" << std::endl;
        std::cout << "[PHYSICS] Expected fit slope: ~0.25" << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();

        Simulations::generate_figure_3();

        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        summary.push_back("Figure 3 | Completed | " + DataOutput::format_duration(duration));

        std::cout << "[SUCCESS] Figure 3 completed in " << DataOutput::format_duration(duration) << std::endl;
        std::cout << std::endl;

        // Generate convergence analysis
        std::cout << "[ANALYSIS] Performing convergence analysis" << std::endl;
        t_start = std::chrono::high_resolution_clock::now();

        std::cout << "[CONVERGENCE] Generating convergence data for Figure 2...\n";
        auto conv_fig2 = Simulations::generate_convergence_analysis();
        std::cout << "[SUCCESS] Convergence data generated.\n";

        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(t_end - t_start).count();
        summary.push_back("Convergence Analysis | Completed | " + DataOutput::format_duration(duration));

        std::cout << "[SUCCESS] Convergence analysis completed in " << DataOutput::format_duration(duration) << std::endl;
        std::cout << std::endl;

        // Generate diagnostic data
        std::cout << "[ANALYSIS] Generating comprehensive diagnostic data" << std::endl;
        t_start = std::chrono::high_resolution_clock::now();
        std::cout << "[DIAGNOSTICS] Generating flux diagnostics...\n";
        auto flux_data = Simulations::diagnose_flux();
        std::cout << "[SUCCESS] Flux diagnostics completed.\n";

        std::cout << "[DIAGNOSTICS] Generating realization data diagnostics...\n";
        auto realization_data = Simulations::diagnose_realization_data();
        std::cout << "[SUCCESS] Realization data diagnostics completed.\n";

        for (int N : N_values) {
            Simulations::diagnose_matrix_distribution(N);
        }

        // Generate flux comparison data
        std::cout << "[ANALYSIS] Generating flux comparison data" << std::endl;
        t_start = std::chrono::high_resolution_clock::now();
        Simulations::generate_flux_comparison_data();
        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(t_end - t_start).count();
        summary.push_back("Flux Comparison | Completed | " + DataOutput::format_duration(duration));
        std::cout << "[SUCCESS] Flux comparison data completed in " << DataOutput::format_duration(duration) << std::endl;
        std::cout << std::endl;


        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(t_end - t_start).count();
        summary.push_back("Diagnostic Data | Completed | " + DataOutput::format_duration(duration));

        std::cout << "[SUCCESS] Diagnostic data generation completed in " << DataOutput::format_duration(duration) << std::endl;
        std::cout << std::endl;

        // Calculate total time
        auto end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end - start).count();

        // Display summary
        std::cout << "======================================================================" << std::endl;
        std::cout << "SIMULATION COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << "Total execution time: " << DataOutput::format_duration(total_duration) << std::endl;
        std::cout << std::endl;

        std::cout << "Task Execution Summary:" << std::endl;
        std::cout << "+------------------------------------+------------+------------------+" << std::endl;
        std::cout << "| Task                               | Status     | Execution Time   |" << std::endl;
        std::cout << "+------------------------------------+------------+------------------+" << std::endl;

        for (const auto& entry : summary) {
            size_t first_pipe = entry.find(" | ");
            size_t second_pipe = entry.find(" | ", first_pipe + 3);

            if (first_pipe != std::string::npos && second_pipe != std::string::npos) {
                std::string task = entry.substr(0, first_pipe);
                std::string status = entry.substr(first_pipe + 3, second_pipe - first_pipe - 3);
                std::string time = entry.substr(second_pipe + 3);

                std::cout << "| " << std::setw(34) << std::left << task
                         << " | " << std::setw(10) << status
                         << " | " << std::setw(16) << time << " |" << std::endl;
            }
        }

        std::cout << "+------------------------------------+------------+------------------+" << std::endl;
        std::cout << std::endl;

        // Physics validation summary with ASCII-safe symbols
        std::cout << "Physics Validation Summary:" << std::endl;
        std::cout << "- Matrix Generation: Exact Haar measure with determinant accuracy validation" << std::endl;
        std::cout << "- Oscillation Probabilities: Corrected equations (4.18 and 4.20) implemented" << std::endl;
        std::cout << "- Coupling Generation: Proper Equation 3.3 matrix-vector multiplication" << std::endl;
        std::cout << "- Expected Results (ASCII-safe notation):" << std::endl;
        std::cout << "  * N=2: P_gg ≈ 0.625, P_eg ≈ 0.25" << std::endl;
        std::cout << "  * Large N: P_gg ≈ 1/N scaling behavior" << std::endl;
        std::cout << "  * Figure 3: g_gamma_50(N) ~ N^(1/4) scaling law" << std::endl;
        std::cout << "  * Matrix errors: Determinant and orthogonality < " << std::scientific << Constants::MATRIX_TOLERANCE << std::endl;
        std::cout << std::endl;

        std::cout << "Next Steps:" << std::endl;
        std::cout << "1. Run the Python plotting script: ./plot" << std::endl;
        std::cout << "2. Check physics validation results in diagnostic outputs" << std::endl;
        std::cout << "3. Verify Figure 3 scaling law passes with R² > 0.95" << std::endl;
        std::cout << "4. Use separated data files for detailed statistical analysis" << std::endl;
        std::cout << std::endl;

        std::cout << "======================================================================" << std::endl;
        std::cout << "ALP ANARCHY SIMULATION PIPELINE COMPLETED" << std::endl;
        std::cout << "All data files available in: " << Constants::DATA_DIR << std::endl;
        std::cout << "======================================================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << std::endl;
        std::cerr << "[ERROR] Simulation failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
