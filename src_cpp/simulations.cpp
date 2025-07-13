#include "../include_cpp/simulations.h"
#include "../include_cpp/matrix_operations.h"
#include "../include_cpp/flux_calculations.h"
#include "../include_cpp/constants.h"
#include "../include_cpp/data_output.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <sstream>

namespace Simulations {
    // Utility to normalize directory path (remove trailing slashes)
    std::string normalize_path(const std::string& path) {
        std::string normalized = path;
        while (!normalized.empty() && (normalized.back() == '/' || normalized.back() == '\\')) {
            normalized.pop_back();
        }
        return normalized;
    }

    RealizationResult single_realization(int N, long double g_e, long double g_gamma, long double phi_max) {
        thread_local std::mt19937 gen(std::random_device{}());
        Eigen::MatrixXd U_gamma = MatrixOperations::random_so_n(N);
        Eigen::MatrixXd U_e = MatrixOperations::random_so_n(N);
        Eigen::VectorXd u_i0_gamma = U_gamma.col(0);
        Eigen::VectorXd u_i0_e = U_e.col(0);
        long double p_gg = MatrixOperations::p_gamma_gamma(u_i0_gamma * g_gamma);
        long double p_eg = MatrixOperations::p_e_gamma(u_i0_e * g_gamma, u_i0_gamma * g_gamma);
        long double phi_osc = p_gg * FluxCalculations::phi_p(g_gamma) + p_eg * (FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e));
        return {phi_osc < phi_max, p_gg, p_eg, phi_osc, u_i0_gamma, u_i0_e};
    }

    std::vector<std::vector<long double>> generate_figure_2(int N) {
        long double g_e_range = Constants::G_E_LOG_MAX - Constants::G_E_LOG_MIN;
        long double g_gamma_range = Constants::G_GAMMA_LOG_MAX - Constants::G_GAMMA_LOG_MIN;
        int g_e_steps = static_cast<int>(g_e_range * static_cast<long double>(Constants::G_E_POINTS_PER_DECADE));
        int g_gamma_steps = static_cast<int>(g_gamma_range * static_cast<long double>(Constants::G_GAMMA_POINTS_PER_DECADE));
        long double g_e_step = g_e_range / static_cast<long double>(g_e_steps);
        long double g_gamma_step = g_gamma_range / static_cast<long double>(g_gamma_steps);
        int total_steps = g_e_steps * g_gamma_steps;
        std::vector<std::vector<long double>> data;

        if (g_e_step <= 0 || !std::isfinite(g_e_step)) {
            std::cerr << "Diagnostic: Invalid g_e_step in generate_figure_2 (N=" << N << "): " << g_e_step << "\n";
            throw std::runtime_error("Invalid g_e_step");
        }
        if (g_gamma_step <= 0 || !std::isfinite(g_gamma_step)) {
            std::cerr << "Diagnostic: Invalid g_gamma_step in generate_figure_2 (N=" << N << "): " << g_gamma_step << "\n";
            throw std::runtime_error("Invalid g_gamma_step");
        }
        if (total_steps <= 0 || g_e_steps <= 0 || g_gamma_steps <= 0) {
            std::cerr << "Diagnostic: Invalid step counts in generate_figure_2 (N=" << N << "): total_steps=" << total_steps
                      << ", g_e_steps=" << g_e_steps << ", g_gamma_steps=" << g_gamma_steps << "\n";
            throw std::runtime_error("Invalid step counts");
        }

        // Create header row
        std::vector<long double> header_row;
        std::stringstream header_ss;
        header_ss << "log_g_e log_g_gamma W";
        for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
            header_ss << " p_gg_" << r;
        }
        for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
            header_ss << " p_eg_" << r;
        }
        for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
            header_ss << " phi_osc_" << r;
        }
        for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
            for (int k = 0; k < N; ++k) {
                header_ss << " u_i0_gamma_" << r << "_" << k;
            }
        }
        for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
            for (int k = 0; k < N; ++k) {
                header_ss << " u_i0_e_" << r << "_" << k;
            }
        }
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        // Create output directory
        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < g_e_steps; ++i) {
            for (int j = 0; j < g_gamma_steps; ++j) {
                long double log_g_e = Constants::G_E_LOG_MIN + i * g_e_step;
                long double log_g_gamma = Constants::G_GAMMA_LOG_MIN + j * g_gamma_step;
                if (log_g_e > Constants::G_E_LOG_MAX + 1e-10 || log_g_gamma > Constants::G_GAMMA_LOG_MAX + 1e-10) continue;
                long double g_e = std::pow(10.0L, log_g_e);
                long double g_gamma = std::pow(10.0L, log_g_gamma);
                long double g_n1 = FluxCalculations::g_gamma_n1(g_e);
                long double phi_n1 = FluxCalculations::phi_p(g_n1) + FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e);
                long double phi_max = (g_n1 * g_n1 / (g_gamma * g_gamma)) * phi_n1;
                long double success_count = 0.0L;
                std::vector<long double> p_gg_vals, p_eg_vals, phi_osc_vals, u_i0_gamma_vals, u_i0_e_vals;

                for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                    auto result = single_realization(N, g_e, g_gamma, phi_max);
                    success_count += result.success ? 1.0L : 0.0L;
                    p_gg_vals.push_back(result.p_gg);
                    p_eg_vals.push_back(result.p_eg);
                    phi_osc_vals.push_back(result.phi_osc);
                    for (int l = 0; l < N; ++l) {
                        u_i0_gamma_vals.push_back(result.u_i0_gamma(l));
                        u_i0_e_vals.push_back(result.u_i0_e(l));
                    }
                }
                long double W = success_count / Constants::N_REALIZATIONS;
                std::vector<long double> row = {log_g_e, log_g_gamma, W};
                row.insert(row.end(), p_gg_vals.begin(), p_gg_vals.end());
                row.insert(row.end(), p_eg_vals.begin(), p_eg_vals.end());
                row.insert(row.end(), phi_osc_vals.begin(), phi_osc_vals.end());
                row.insert(row.end(), u_i0_gamma_vals.begin(), u_i0_gamma_vals.end());
                row.insert(row.end(), u_i0_e_vals.begin(), u_i0_e_vals.end());

                #pragma omp critical
                {
                    data.push_back(row);
                    if (data.size() % (total_steps / 10) == 0) {
                        int progress = static_cast<int>(data.size() * 100.0 / total_steps);
                        int bar_width = 20;
                        int filled = progress / (100 / bar_width);
                        std::string bar(filled, '=');
                        bar.append(bar_width - filled, ' ');
                        std::cout << "\rFigure 2 (N=" << N << "): [" << bar << "] " << progress << "%" << std::flush;
                    }
                }
            }
        }
        std::cout << "\rFigure 2 (N=" << N << "): [====================] 100%" << std::endl;

        if (data.size() != static_cast<size_t>(total_steps) + 1) {
            std::cerr << "ERROR: Expected " << total_steps << " data rows (plus 1 header), but generated " << (data.size() - 1) << " data rows for N=" << N << "\n";
            throw std::runtime_error("Incorrect number of data rows generated");
        }

        std::string output_file = data_dir + "/figure_2_data_N" + std::to_string(N) + ".txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << total_steps << ", g_e_steps=" << g_e_steps << ", g_gamma_steps=" << g_gamma_steps << ")" << "\n";
        DataOutput::write_figure_2_data(data, output_file, header_ss.str());
        return data;
    }

    std::vector<std::vector<long double>> generate_figure_3() {
        long double g_e = 1e-15;
        int g_gamma_points = Constants::G_GAMMA_POINTS;
        std::vector<std::vector<long double>> data;
        int total_steps = Constants::MAX_N - 1;
        int current_step = 0;

        // Create header row
        std::vector<long double> header_row;
        std::stringstream header_ss;
        header_ss << "N";
        for (int i = 0; i < g_gamma_points; ++i) {
            header_ss << " log_g_50_" << i;
        }
        for (int i = 0; i < g_gamma_points; ++i) {
            for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
                header_ss << " p_gg_" << i << "_" << r;
            }
        }
        for (int i = 0; i < g_gamma_points; ++i) {
            for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
                for (int k = 0; k < Constants::MAX_N; ++k) {
                    header_ss << " u_i0_" << i << "_" << r << "_" << k;
                }
            }
        }
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        // Create output directory
        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        for (int N = 2; N <= Constants::MAX_N; ++N) {
            std::vector<long double> log_g_50_vals(g_gamma_points);
            std::vector<long double> p_gg_vals(g_gamma_points * Constants::N_REALIZATIONS);
            std::vector<long double> u_i0_vals(g_gamma_points * Constants::N_REALIZATIONS * Constants::MAX_N);
            long double log_g_min = Constants::G_GAMMA_LOG_MIN;
            long double log_g_max = Constants::G_GAMMA_LOG_MAX;
            long double g_step = (log_g_max - log_g_min) / (g_gamma_points - 1);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < g_gamma_points; ++i) {
                long double log_g_gamma = log_g_min + i * g_step;
                long double g_gamma = std::pow(10.0L, log_g_gamma);
                long double g_n1 = FluxCalculations::g_gamma_n1(g_e);
                long double phi_n1 = FluxCalculations::phi_p(g_n1) + FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e);
                long double phi_max = (g_n1 * g_n1 / (g_gamma * g_gamma)) * phi_n1;
                long double success_count = 0.0L;
                std::vector<long double> p_gg_temp(Constants::N_REALIZATIONS);
                std::vector<long double> u_i0_temp(Constants::N_REALIZATIONS * N);
                for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                    auto result = single_realization(N, g_e, g_gamma, phi_max);
                    p_gg_temp[k] = result.p_gg;
                    for (int l = 0; l < N; ++l) u_i0_temp[k * N + l] = result.u_i0_gamma(l);
                    success_count += result.success ? 1.0L : 0.0L;
                }
                log_g_50_vals[i] = log_g_gamma;
                for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                    p_gg_vals[i * Constants::N_REALIZATIONS + k] = p_gg_temp[k];
                    for (int l = 0; l < N; ++l) {
                        u_i0_vals[i * Constants::N_REALIZATIONS * Constants::MAX_N + k * Constants::MAX_N + l] = u_i0_temp[k * N + l];
                    }
                }
            }

            std::vector<long double> W_vals(g_gamma_points);
            for (int i = 0; i < g_gamma_points; ++i) {
                long double g_gamma = std::pow(10.0L, log_g_50_vals[i]);
                long double g_n1 = FluxCalculations::g_gamma_n1(g_e);
                long double phi_n1 = FluxCalculations::phi_p(g_n1) + FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e);
                long double phi_max = (g_n1 * g_n1 / (g_gamma * g_gamma)) * phi_n1;
                W_vals[i] = std::count_if(p_gg_vals.begin() + i * Constants::N_REALIZATIONS, p_gg_vals.begin() + (i + 1) * Constants::N_REALIZATIONS,
                                          [phi_max, g_gamma](long double p_gg) {
                                              return p_gg * FluxCalculations::phi_p(g_gamma) < phi_max;
                                          }) / (long double)Constants::N_REALIZATIONS;
            }
            int idx_50 = std::distance(W_vals.begin(), std::min_element(W_vals.begin(), W_vals.end(),
                                      [](long double a, long double b) { return std::abs(a - 0.5) < std::abs(b - 0.5); }));

            std::vector<long double> row = {(long double)N};
            row.insert(row.end(), log_g_50_vals.begin(), log_g_50_vals.end());
            row.insert(row.end(), p_gg_vals.begin(), p_gg_vals.end());
            row.insert(row.end(), u_i0_vals.begin(), u_i0_vals.end());
            data.push_back(row);

            #pragma omp critical
            {
                current_step++;
                if (current_step % (total_steps / 10) == 0) {
                    int progress = static_cast<int>(current_step * 100.0 / total_steps);
                    int bar_width = 20;
                    int filled = progress / (100 / bar_width);
                    std::string bar(filled, '=');
                    bar.append(bar_width - filled, ' ');
                    std::cout << "\rFigure 3: [" << bar << "] " << progress << "%" << std::flush;
                }
            }
        }
        std::cout << "\rFigure 3: [====================] 100%" << std::endl;

        if (data.size() != static_cast<size_t>(total_steps) + 1) {
            std::cerr << "ERROR: Expected " << total_steps << " data rows (plus 1 header), but generated " << (data.size() - 1) << " data rows for Figure 3\n";
            throw std::runtime_error("Incorrect number of data rows generated");
        }

        std::string output_file = data_dir + "/figure_3_data.txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << total_steps << ")" << "\n";
        DataOutput::write_figure_3_data(data, output_file, header_ss.str());
        return data;
    }

    std::vector<std::vector<long double>> generate_convergence_fig2() {
        std::vector<std::vector<long double>> data;
        std::vector<long double> header_row;
        std::stringstream header_ss;
        header_ss << "N W";
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        for (int N = 2; N <= Constants::MAX_N; ++N) {
            long double W_sum = 0.0L;
            for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
                auto result = single_realization(N, std::pow(10.0L, Constants::G_E_LOG_MAX),
                                               std::pow(10.0L, Constants::G_GAMMA_LOG_MAX),
                                               Constants::SOLAR_NU_LIMIT);
                W_sum += result.success ? 1.0L : 0.0L;
            }
            data.push_back(std::vector<long double>{static_cast<long double>(N), W_sum / Constants::N_REALIZATIONS});
            int progress = static_cast<int>((N - 1) * 100.0 / (Constants::MAX_N - 1));
            int bar_width = 20;
            int filled = progress / (100 / bar_width);
            std::string bar(filled, '=');
            bar.append(bar_width - filled, ' ');
            std::cout << "\rConvergence Fig2: [" << bar << "] " << progress << "%" << std::flush;
        }
        std::cout << "\rConvergence Fig2: [====================] 100%" << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        std::string output_file = data_dir + "/convergence_fig2.txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << (Constants::MAX_N - 1) << ")" << "\n";
        DataOutput::write_convergence_fig2_data(data, output_file, header_ss.str());
        return data;
    }

    std::vector<std::vector<long double>> generate_convergence_fig3() {
        std::vector<std::vector<long double>> data;
        std::vector<long double> header_row;
        std::stringstream header_ss;
        header_ss << "N W";
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        for (int N = 2; N <= Constants::MAX_N; ++N) {
            long double W_sum = 0.0L;
            for (int r = 0; r < Constants::N_REALIZATIONS; ++r) {
                auto result = single_realization(N, Constants::G_GAMMA_G_E_LIMIT / std::pow(10.0L, Constants::G_GAMMA_LOG_MAX),
                                               std::pow(10.0L, Constants::G_GAMMA_LOG_MAX),
                                               Constants::SOLAR_NU_LIMIT);
                W_sum += result.success ? 1.0L : 0.0L;
            }
            data.push_back(std::vector<long double>{static_cast<long double>(N), W_sum / Constants::N_REALIZATIONS});
            int progress = static_cast<int>((N - 1) * 100.0 / (Constants::MAX_N - 1));
            int bar_width = 20;
            int filled = progress / (100 / bar_width);
            std::string bar(filled, '=');
            bar.append(bar_width - filled, ' ');
            std::cout << "\rConvergence Fig3: [" << bar << "] " << progress << "%" << std::flush;
        }
        std::cout << "\rConvergence Fig3: [====================] 100%" << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        std::string output_file = data_dir + "/convergence_fig3.txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << (Constants::MAX_N - 1) << ")" << "\n";
        DataOutput::write_convergence_fig3_data(data, output_file, header_ss.str());
        return data;
    }

    std::vector<std::vector<long double>> diagnose_flux() {
        std::vector<std::vector<long double>> data;
        std::vector<long double> header_row;
        std::stringstream header_ss;
        header_ss << "log_g_e phi_osc phi_b phi_c";
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        int points = 100;
        for (int i = 0; i < points; ++i) {
            long double log_g_e = -15.0L + i * (4.0L / points);
            long double g_e = std::pow(10.0L, log_g_e);
            long double phi_osc = FluxCalculations::phi_p(std::pow(10.0L, Constants::G_GAMMA_LOG_MAX)) +
                                 FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e);
            data.push_back(std::vector<long double>{log_g_e, phi_osc, FluxCalculations::phi_b(g_e), FluxCalculations::phi_c(g_e)});
            if (i % (points / 10) == 0) {
                int progress = static_cast<int>(i * 100.0 / points);
                int bar_width = 20;
                int filled = progress / (100 / bar_width);
                std::string bar(filled, '=');
                bar.append(bar_width - filled, ' ');
                std::cout << "\rDiagnostics Flux: [" << bar << "] " << progress << "%" << std::flush;
            }
        }
        std::cout << "\rDiagnostics Flux: [====================] 100%" << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        std::string output_file = data_dir + "/diagnostics_flux.txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << points << ")" << "\n";
        DataOutput::write_diagnostics_flux_data(data, output_file, header_ss.str());
        return data;
    }

    std::vector<std::vector<long double>> diagnose_realization_data() {
        std::vector<std::vector<long double>> data;
        std::vector<long double> header_row;
        std::stringstream header_ss;
        header_ss << "N p_gg_mean p_gg_sem";
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        for (int N = 2; N <= Constants::MAX_N; ++N) {
            long double p_gg_sum = 0.0L, p_gg_sum_sq = 0.0L;
            for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                auto result = single_realization(N, std::pow(10.0L, Constants::G_E_LOG_MAX),
                                               std::pow(10.0L, Constants::G_GAMMA_LOG_MAX),
                                               Constants::SOLAR_NU_LIMIT);
                p_gg_sum += result.p_gg;
                p_gg_sum_sq += result.p_gg * result.p_gg;
            }
            long double p_gg_mean = p_gg_sum / Constants::N_REALIZATIONS;
            long double p_gg_sem = std::sqrt((p_gg_sum_sq / Constants::N_REALIZATIONS - p_gg_mean * p_gg_mean) / Constants::N_REALIZATIONS);
            data.push_back({(long double)N, p_gg_mean, p_gg_sem});
            int progress = static_cast<int>((N - 1) * 100.0 / (Constants::MAX_N - 1));
            int bar_width = 20;
            int filled = progress / (100 / bar_width);
            std::string bar(filled, '=');
            bar.append(bar_width - filled, ' ');
            std::cout << "\rDiagnostics Realization: [" << bar << "] " << progress << "%" << std::flush;
        }
        std::cout << "\rDiagnostics Realization: [====================] 100%" << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        std::string output_file = data_dir + "/diagnostics_realization.txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << (Constants::MAX_N - 1) << ")" << "\n";
        DataOutput::write_diagnostics_realization_data(data, output_file, header_ss.str());
        return data;
    }

    std::vector<std::vector<long double>> diagnose_matrix_distribution(int N) {
        std::vector<std::vector<long double>> data;
        std::vector<long double> header_row;
        std::stringstream header_ss;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                header_ss << " U_" << i << "_" << j;
            }
        }
        header_ss << " determinant";
        header_row.push_back(std::numeric_limits<long double>::quiet_NaN());
        data.push_back(header_row);

        for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
            Eigen::MatrixXd U = MatrixOperations::random_so_n(N);
            std::vector<long double> row;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    row.push_back(U(i, j));
                }
            }
            row.push_back(U.determinant());
            data.push_back(row);
            if ((k + 1) % (Constants::N_REALIZATIONS / 10) == 0) {
                int progress = static_cast<int>((k + 1) * 100.0 / Constants::N_REALIZATIONS);
                int bar_width = 20;
                int filled = progress / (100 / bar_width);
                std::string bar(filled, '=');
                bar.append(bar_width - filled, ' ');
                std::cout << "\rMatrix distribution (N=" << N << "): [" << bar << "] " << progress << "%" << std::flush;
            }
        }
        std::cout << "\rMatrix distribution (N=" << N << "): [====================] 100%" << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        if (!std::filesystem::create_directories(data_dir)) {
            if (!std::filesystem::exists(data_dir)) {
                std::cerr << "ERROR: Failed to create directory: " << data_dir << "\n";
                throw std::runtime_error("Failed to create data directory");
            }
        }

        std::string output_file = data_dir + "/matrix_distributions_N" + std::to_string(N) + ".txt";
        std::cerr << "Attempting to write to: " << output_file << "\n";
        std::cerr << "DEBUG: Total data rows written: " << (data.size() - 1) << " (expected " << Constants::N_REALIZATIONS << ")" << "\n";
        DataOutput::write_matrix_distribution_data(data, output_file, header_ss.str());
        return data;
    }
}
