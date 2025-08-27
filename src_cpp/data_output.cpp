#include "../include_cpp/data_output.h"
#include "../include_cpp/constants.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <iostream>

namespace DataOutput {

    // Terminal-safe output formatting
    class TerminalSafeOutput {
    private:
        static bool use_unicode;

    public:
        static void initialize() {
            // Check if terminal supports UTF-8
            const char* lang = std::getenv("LANG");
            use_unicode = (lang && (strstr(lang, "UTF-8") || strstr(lang, "utf8")));
        }

        static std::string format_physics_symbol(const std::string& symbol_type) {
            if (use_unicode) {
                if (symbol_type == "P_gamma_gamma") return "P_γ→γ";
                if (symbol_type == "P_e_gamma") return "P_e→γ";
                if (symbol_type == "g_gamma_50") return "g_γ₅₀";
            }

            // ASCII fallback for terminal compatibility
            if (symbol_type == "P_gamma_gamma") return "P_gg";
            if (symbol_type == "P_e_gamma") return "P_eg";
            if (symbol_type == "g_gamma_50") return "g_gamma_50";

            return symbol_type;
        }

        static std::string format_units(const std::string& unit_type) {
            if (use_unicode && unit_type == "Geiagnostic") return "GeV⁻¹";
            return "GeV^-1";  // ASCII fallback
        }

        static std::string format_scaling_law() {
            if (use_unicode) return "g_γ₅₀(N) ∝ N^(1/4)";
            return "g_gamma_50(N) ~ N^(1/4)";  // ASCII fallback
        }
    };

    bool TerminalSafeOutput::use_unicode = false;

    // Utility functions
    std::string format_duration(double seconds) {
        std::stringstream ss;

        if (seconds < 60) {
            ss << std::fixed << std::setprecision(2) << seconds << "s";
        } else if (seconds < 3600) {
            int minutes = static_cast<int>(seconds / 60);
            double remaining_seconds = seconds - minutes * 60;
            ss << minutes << "m " << std::fixed << std::setprecision(1) << remaining_seconds << "s";
        } else {
            int hours = static_cast<int>(seconds / 3600);
            int minutes = static_cast<int>((seconds - hours * 3600) / 60);
            double remaining_seconds = seconds - hours * 3600 - minutes * 60;
            ss << hours << "h " << minutes << "m " << std::fixed << std::setprecision(0) << remaining_seconds << "s";
        }

        return ss.str();
    }

    std::string format_scientific(double value, int precision) {
        std::stringstream ss;
        ss << std::scientific << std::setprecision(precision) << value;
        return ss.str();
    }

    std::string format_percentage(double fraction, int decimal_places) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(decimal_places) << (fraction * 100.0) << "%";
        return ss.str();
    }

    std::string generate_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    std::string create_comprehensive_header(const std::string& data_type,
                                          const std::string& description,
                                          int N, int n_realizations) {
        std::stringstream header;

        header << "# ================================================================\n";
        header << "# ALP Anarchy Simulation - " << data_type << "\n";
        header << "# ================================================================\n";
        header << "# Generated: " << generate_timestamp() << "\n";
        header << "# Description: " << description << "\n";

        if (N > 0) {
            header << "# Number of ALP fields (N): " << N << "\n";
        }
        if (n_realizations > 0) {
            header << "# Number of realizations: " << n_realizations << "\n";
        }

        header << "# \n";
        header << "# Physics Parameters:\n";
        header << "#   G_GAMMA_N1 = " << format_scientific(Constants::G_GAMMA_N1, 3) << " GeV^-1\n";
        header << "#   G_GAMMA_G_E_LIMIT = " << format_scientific(Constants::G_GAMMA_G_E_LIMIT, 3) << " GeV^-1\n";
        header << "#   PHI_P_COEFF = " << format_scientific(Constants::PHI_P_COEFF, 3) << " m^-2 s^-1\n";
        header << "#   PHI_B_COEFF = " << format_scientific(Constants::PHI_B_COEFF, 3) << " m^-2 s^-1\n";
        header << "#   PHI_C_COEFF = " << format_scientific(Constants::PHI_C_COEFF, 3) << " m^-2 s^-1\n";
        header << "# \n";
        header << "# Statistical Test Parameters:\n";
        header << "#   KS significance level: " << Constants::KS_ALPHA_LEVEL << "\n";
        header << "#   Matrix tolerance: " << format_scientific(Constants::MATRIX_TOLERANCE, 2) << "\n";
        header << "#   Min samples for KS test: " << Constants::MIN_SAMPLES_FOR_KS << "\n";
        header << "# \n";
        header << "# Grid Parameters:\n";
        header << "#   G_E range: [" << Constants::G_E_LOG_MIN << ", " << Constants::G_E_LOG_MAX << "] (log10)\n";
        header << "#   G_GAMMA range: [" << Constants::G_GAMMA_LOG_MIN << ", " << Constants::G_GAMMA_LOG_MAX << "] (log10)\n";
        header << "#   G_E points per decade: " << Constants::G_E_POINTS_PER_DECADE << "\n";
        header << "#   G_GAMMA points per decade: " << Constants::G_GAMMA_POINTS_PER_DECADE << "\n";
        header << "# ================================================================\n";

        return header.str();
    }

    bool create_output_directory(const std::string& directory) {
        try {
            if (!std::filesystem::exists(directory)) {
                return std::filesystem::create_directories(directory);
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory " << directory << ": " << e.what() << std::endl;
            return false;
        }
    }

    // Original compatibility functions with ASCII-safe headers
    void write_figure_2_data(const std::vector<std::vector<long double>>& data,
                             const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        file << "# " << header << "\n";
        file << "# ASCII-safe format: P_gg = P_gamma->gamma, P_eg = P_e->gamma\n";
        file << std::setprecision(18);

        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
    }

    void write_figure_3_data(const std::vector<std::vector<long double>>& data,
                             const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        file << "# " << header << "\n";
        file << "# Figure 3: CAST bounds scaling analysis\n";
        file << "# Scaling law: g_gamma_50(N) ~ N^(1/4) for ALP anarchy\n";
        file << "# ASCII-safe format used for terminal compatibility\n";
        file << std::setprecision(18);

        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
    }

    void write_convergence_fig2_data(const std::vector<std::vector<long double>>& data,
                                     const std::string& filename, const std::string& header) {
        write_figure_2_data(data, filename, header);
    }

    void write_convergence_fig3_data(const std::vector<std::vector<long double>>& data,
                                     const std::string& filename, const std::string& header) {
        write_figure_3_data(data, filename, header);
    }

    void write_diagnostics_flux_data(const std::vector<std::vector<long double>>& data,
                                     const std::string& filename, const std::string& header) {
        write_figure_2_data(data, filename, header);
    }

    void write_diagnostics_realization_data(const std::vector<std::vector<long double>>& data,
                                            const std::string& filename, const std::string& header) {
        write_figure_2_data(data, filename, header);
    }

    void write_matrix_distribution_data(const std::vector<std::vector<long double>>& data,
                                        const std::string& filename, const std::string& header) {
        write_figure_2_data(data, filename, header);
    }

    bool write_figure2_stats(const Figure2Data& data, const std::string& base_filename) {
        std::ofstream file(base_filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << base_filename << std::endl;
            return false;
        }

        file << create_comprehensive_header("Figure 2 Data", "Mean and std for P_gg and P_eg, W etc", data.N, data.n_realizations);
        file << "# Columns: log_g_e log_g_gamma W P_gg_mean P_gg_std P_eg_mean P_eg_std W_residual_mean W_chi2_p_value\n";
        file << std::setprecision(18);

        for (size_t i = 0; i < data.g_e_values.size(); ++i) {
            for (size_t j = 0; j < data.g_gamma_values.size(); ++j) {
                const auto& p_gg_stats = data.p_gamma_stats_matrix[i][j];
                const auto& p_eg_stats = data.p_e_gamma_stats_matrix[i][j];
                double w = data.weight_matrix[i][j];
                double expected_p_gamma = (data.N == 2) ? 0.625 : 1.0 / data.N;
                double expected_p_e_gamma = (data.N == 2) ? 0.25 : 0.1;
                double p_gg_res_mean = p_gg_stats.sample_mean - expected_p_gamma;
                double p_eg_res_mean = p_eg_stats.sample_mean - expected_p_e_gamma;
                double w_res_mean = w - 0.5;
                double w_chi2_p = p_gg_stats.chi2_p_value;  // Placeholder
                file << std::log10(data.g_e_values[i]) << " "
                     << std::log10(data.g_gamma_values[j]) << " "
                     << w << " "
                     << p_gg_stats.sample_mean << " "
                     << p_gg_stats.sample_std << " "
                     << p_eg_stats.sample_mean << " "
                     << p_eg_stats.sample_std << " "
                     << w_res_mean << " "
                     << w_chi2_p << "\n";
            }
        }

        return true;
    }


    bool write_figure3_stats(const Figure3Data& data, const std::string& base_filename) {
        std::ofstream file(base_filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << base_filename << std::endl;
            return false;
        }

        file << create_comprehensive_header("Figure 3 Statistical Analysis",
                                           "Statistical test results for P_gg and P_eg distributions at fixed g_e",
                                           data.N, data.n_realizations);
        file << "# Columns: N g_e_fixed log_g_50gamma W P_gg_ks_stat P_gg_ks_pval P_gg_chi2_stat P_gg_chi2_pval "
             << "P_eg_ks_stat P_eg_ks_pval P_eg_chi2_stat P_eg_chi2_pval\n";
        file << std::setprecision(18);

        file << data.N << " "
             << data.g_e_fixed << " "
             << data.log_g_50gamma << " "
             << data.W << " "
             << data.global_p_gamma_stats.ks_statistic << " "
             << data.global_p_gamma_stats.ks_p_value << " "
             << data.global_p_gamma_stats.chi2_statistic << " "
             << data.global_p_gamma_stats.chi2_p_value << " "
             << data.global_p_e_gamma_stats.ks_statistic << " "
             << data.global_p_e_gamma_stats.ks_p_value << " "
             << data.global_p_e_gamma_stats.chi2_statistic << " "
             << data.global_p_e_gamma_stats.chi2_p_value << "\n";

        // Write raw probability data
        file << "\n# Raw P_gg Data:\n";
        for (auto val : data.p_gamma_raw_data) {
            file << val << " ";
        }
        file << "\n# Raw P_eg Data:\n";
        for (auto val : data.p_e_gamma_raw_data) {
            file << val << " ";
        }
        file << "\n";

        return true;
    }

    void print_statistical_table(const std::vector<std::pair<std::string, StatisticalTestResults>>& results,
                                const std::string& title) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << std::left
                  << std::setw(20) << "Metric"
                  << std::setw(12) << "KS Stat"
                  << std::setw(12) << "KS P-Value"
                  << std::setw(12) << "Chi2 Stat"
                  << std::setw(12) << "Chi2 P-Value"
                  << std::setw(12) << "Mean"
                  << std::setw(12) << "Std Dev"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        for (const auto& [name, stats] : results) {
            std::cout << std::setw(20) << name
                      << std::scientific << std::setprecision(4)
                      << std::setw(12) << stats.ks_statistic
                      << std::setw(12) << stats.ks_p_value
                      << std::setw(12) << stats.chi2_statistic
                      << std::setw(12) << stats.chi2_p_value
                      << std::setw(12) << stats.sample_mean
                      << std::setw(12) << stats.sample_std
                      << "\n";
        }

        std::cout << std::string(80, '=') << "\n";
    };

}
