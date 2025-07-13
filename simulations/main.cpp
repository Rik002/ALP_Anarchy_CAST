#include "../include_cpp/simulations.h"
#include "../include_cpp/constants.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "ALP Anarchy CAST Simulation Pipeline\n";
    std::vector<std::string> summary;

    try {
        Constants::load_constants("../config/constants.dat");
        std::cout << "Loaded constants from ../config/constants.dat\n";
        std::cout << "Output directory: " << Constants::DATA_DIR << "\n";

        int N_values[] = {2, 10, 30};
        for (int N : N_values) {
            std::cout << "Generating Figure 2 for N=" << N << "...\n";
            auto t_start = std::chrono::high_resolution_clock::now();
            Simulations::generate_figure_2(N);
            auto t_end = std::chrono::high_resolution_clock::now();
            long double duration = std::chrono::duration<long double>(t_end - t_start).count();
            summary.push_back("Figure 2 (N=" + std::to_string(N) + ") | Completed | " + std::to_string(duration) + " seconds");
            std::cout << "Figure 2 (N=" << N << ") completed in " << duration << " seconds\n";
        }

        std::cout << "Generating Figure 3...\n";
        auto t_start = std::chrono::high_resolution_clock::now();
        Simulations::generate_figure_3();
        auto t_end = std::chrono::high_resolution_clock::now();
        long double duration = std::chrono::duration<long double>(t_end - t_start).count();
        summary.push_back("Figure 3 | Completed | " + std::to_string(duration) + " seconds");
        std::cout << "Figure 3 completed in " << duration << " seconds\n";

        std::cout << "Generating convergence data...\n";
        t_start = std::chrono::high_resolution_clock::now();
        Simulations::generate_convergence_fig2();
        Simulations::generate_convergence_fig3();
        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<long double>(t_end - t_start).count();
        summary.push_back("Convergence Data | Completed | " + std::to_string(duration) + " seconds");
        std::cout << "Convergence data completed in " << duration << " seconds\n";

        std::cout << "Generating diagnostic data...\n";
        t_start = std::chrono::high_resolution_clock::now();
        Simulations::diagnose_flux();
        Simulations::diagnose_realization_data();
        for (int N : N_values) {
            std::cout << "Generating matrix distribution for N=" << N << "...\n";
            Simulations::diagnose_matrix_distribution(N);
        }
        t_end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<long double>(t_end - t_start).count();
        summary.push_back("Diagnostic Data | Completed | " + std::to_string(duration) + " seconds");
        std::cout << "Diagnostic data completed in " << duration << " seconds\n";

        auto end = std::chrono::high_resolution_clock::now();
        long double total_duration = std::chrono::duration<long double>(end - start).count();
        std::cout << "\nSimulation Summary:\n";
        std::cout << "+-----------------------------+------------+----------------+\n";
        std::cout << "| Task                        | Status     | Time (seconds) |\n";
        std::cout << "+-----------------------------+------------+----------------+\n";
        for (const auto& entry : summary) {
            size_t first = entry.find(" | ");
            size_t second = entry.find(" | ", first + 3);
            std::string task = entry.substr(0, first);
            std::string status = entry.substr(first + 3, second - first - 3);
            std::string time = entry.substr(second + 3);
            std::cout << "| " << std::setw(27) << std::left << task
                      << " | " << std::setw(10) << status
                      << " | " << std::setw(14) << time << " |\n";
        }
        std::cout << "+-----------------------------+------------+----------------+\n";
        std::cout << "Total simulation time: " << total_duration << " seconds\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        summary.push_back("Pipeline | Failed | N/A");
        std::cout << "\nSimulation Summary:\n";
        std::cout << "+-----------------------------+------------+----------------+\n";
        std::cout << "| Task                        | Status     | Time (seconds) |\n";
        std::cout << "+-----------------------------+------------+----------------+\n";
        for (const auto& entry : summary) {
            size_t first = entry.find(" | ");
            size_t second = entry.find(" | ", first + 3);
            std::string task = entry.substr(0, first);
            std::string status = entry.substr(first + 3, second - first - 3);
            std::string time = entry.substr(second + 3);
            std::cout << "| " << std::setw(27) << std::left << task
                      << " | " << std::setw(10) << status
                      << " | " << std::setw(14) << time << " |\n";
        }
        std::cout << "+-----------------------------+------------+----------------+\n";
        return 1;
    }
    return 0;
}
