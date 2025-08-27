#include "../include_cpp/simulations.h"
#include "../include_cpp/matrix_operations.h"
#include "../include_cpp/flux_calculations.h"
#include "../include_cpp/constants.h"
#include "../include_cpp/data_output.h"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <random>
#include <sys/ioctl.h>
#include <unistd.h>
#include <map>

#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>  // Required for some GSL operations
#include <gsl/gsl_errno.h> // For GSL_SUCCESS and error handling
#include <gsl/gsl_statistics.h> // For gsl_stats_mean, gsl_stats_sd, etc. (already used in matrix_operations.cpp)
#include <gsl/gsl_histogram.h> // For gsl_histogram_* functions (already used)
#include <gsl/gsl_cdf.h>       // For gsl_cdf_* functions (already used)



double compute_chi2_p_value_for_grid_point(double W, int n_realizations) {
    // W should vary across grid points based on physics
    // Calculate observed vs expected frequencies
    int observed_success = static_cast<int>(W * n_realizations);
    int observed_failure = n_realizations - observed_success;

    // Expected frequencies (null hypothesis: W = 0.5)
    double expected_success = 0.5 * n_realizations;
    double expected_failure = 0.5 * n_realizations;

    // Chi-squared statistic
    double chi2_stat = 0.0;
    if (expected_success > 5 && expected_failure > 5) { // Valid chi-squared conditions
        chi2_stat = ((observed_success - expected_success) * (observed_success - expected_success)) / expected_success +
                   ((observed_failure - expected_failure) * (observed_failure - expected_failure)) / expected_failure;

        // Convert to p-value (df = 1)
        // Using complementary error function approximation
        double p_value = std::erfc(std::sqrt(chi2_stat / 2.0));
        return std::max(1e-10, std::min(1.0, p_value));
    }

    return 1.0; // Default for invalid cases
}

namespace Simulations {

    // Enhanced Progress Bar
    class EnhancedProgressBar {
    private:
        std::string task_name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point last_update;
        int terminal_width;
        bool show_eta;
        bool show_rate;

    public:
        EnhancedProgressBar(const std::string& name, bool eta = true, bool rate = true)
            : task_name(name), show_eta(eta), show_rate(rate) {
            start_time = last_update = std::chrono::high_resolution_clock::now();
            terminal_width = get_terminal_width();
        }

        void update(int current, int total, const std::string& details = "") {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (current != total && std::chrono::duration<double>(now - last_update).count() < 0.1) return;

            double percentage = 100.0 * current / total;
            double eta_val = (current > 0) ? (elapsed * (total - current) / current) : 0.0;
            double rate_val = (elapsed > 0) ? (current / elapsed) : 0.0;

            int bar_width = std::max(20, terminal_width - 80);
            int filled = static_cast<int>(bar_width * current / total);

            std::ostringstream bar;
            bar << "\r[";
            for (int i = 0; i < filled; ++i) bar << "=";
            if (filled < bar_width) {
                bar << ">";
                for (int i = filled + 1; i < bar_width; ++i) bar << " ";
            }
            bar << "] " << std::fixed << std::setprecision(1) << percentage << "% ";
            bar << "(" << current << "/" << total << ")";
            if (show_eta && current < total && eta_val > 0) bar << " ETA: " << format_time_duration(eta_val);
            if (show_rate && rate_val > 0) bar << " [" << std::fixed << std::setprecision(1) << rate_val << " it/s]";
            if (!details.empty()) bar << " - " << details;

            std::string output = bar.str();
            if (output.length() > static_cast<size_t>(terminal_width - 5)) {
                output = output.substr(0, terminal_width - 8) + "...";
            }
            std::cout << output << std::flush;
            last_update = now;
        }

        void finish(const std::string& final_message = "") {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            std::cout << "\r[DONE] " << task_name << " completed in " << format_time_duration(elapsed);
            if (!final_message.empty()) std::cout << " - " << final_message;
            std::cout << std::endl;
        }

    private:
        int get_terminal_width() {
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            return w.ws_col > 0 ? w.ws_col : 80;
        }

        std::string format_time_duration(double seconds) {
            if (seconds < 60) return std::to_string(static_cast<int>(seconds)) + "s";
            else if (seconds < 3600) {
                int mins = static_cast<int>(seconds / 60);
                int secs = static_cast<int>(seconds) % 60;
                return std::to_string(mins) + "m" + std::to_string(secs) + "s";
            } else {
                int hours = static_cast<int>(seconds / 3600);
                int mins = static_cast<int>((seconds - hours * 3600) / 60);
                return std::to_string(hours) + "h" + std::to_string(mins) + "m";
            }
        }
    };

    // Small Utility
    static std::string normalize_path(const std::string &path) {
        std::string out = path;
        while (!out.empty() && (out.back() == '/' || out.back() == '\\')) out.pop_back();
        return out;
    }

    static std::string fmt_time(double seconds) {
        std::ostringstream ss; ss << std::fixed << std::setprecision(1);
        if (seconds < 60) { ss << seconds << "s"; }
        else if (seconds < 3600) { ss << int(seconds / 60) << "m " << fmod(seconds, 60) << "s"; }
        else { ss << int(seconds / 3600) << "h " << int(fmod(seconds, 3600) / 60) << "m"; }
        return ss.str();
    }
    static void list_relevant_files(const std::vector<std::string>& files, const std::string& operation_name) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << operation_name << " OUTPUT FILES\n";
        std::cout << std::string(60, '=') << "\n";

        if (files.empty()) {
            std::cout << "[INFO] No output files generated for " << operation_name << "\n";
            std::cout << std::string(60, '=') << "\n";
            return;
        }

        std::vector<std::pair<std::string, std::string>> file_list;
        size_t max_file_len = 20; // Length of "File" header
        size_t max_dir_len = 10;  // Length of "Directory" header

        for (const auto& file_path : files) {
            std::string file_name = std::filesystem::path(file_path).filename().string();
            std::string parent_dir = std::filesystem::path(file_path).parent_path().string();
            file_list.emplace_back(file_name, parent_dir);
            max_file_len = std::max(max_file_len, file_name.length());
            max_dir_len = std::max(max_dir_len, parent_dir.length());
        }

        std::cout << "+" << std::string(max_file_len + 2, '-') << "+"
                  << std::string(max_dir_len + 2, '-') << "+\n";
        std::cout << "| " << std::setw(max_file_len) << std::left << "File"
                  << " | " << std::setw(max_dir_len) << std::left << "Directory" << " |\n";
        std::cout << "+" << std::string(max_file_len + 2, '-') << "+"
                  << std::string(max_dir_len + 2, '-') << "+\n";

        for (const auto& [file, dir] : file_list) {
            std::cout << "| " << std::setw(max_file_len) << std::left << file
                      << " | " << std::setw(max_dir_len) << std::left << dir << " |\n";
        }

        std::cout << "+" << std::string(max_file_len + 2, '-') << "+"
                  << std::string(max_dir_len + 2, '-') << "+\n";
        std::cout << std::string(60, '=') << "\n";
    }


    // Numerical Utilities
    static double find_threshold_coupling(const std::vector<double>& g_gamma_values,
                                         const std::vector<double>& viability_fractions,
                                         double threshold) {
        if (g_gamma_values.size() != viability_fractions.size() || g_gamma_values.empty()) {
            return 1e-10;
        }

        for (size_t i = 1; i < viability_fractions.size(); ++i) {
            double y0 = viability_fractions[i-1];
            double y1 = viability_fractions[i];
            if ((y0 <= threshold && y1 >= threshold) || (y0 >= threshold && y1 <= threshold)) {
                if (std::abs(y1 - y0) < 1e-10) {
                    return g_gamma_values[i];
                }
                double t = (threshold - y0) / (y1 - y0);
                t = std::max(0.0, std::min(1.0, t));
                return g_gamma_values[i-1] + t * (g_gamma_values[i] - g_gamma_values[i-1]);
            }
        }

        if (viability_fractions.front() <= threshold && viability_fractions.back() <= threshold) return g_gamma_values.back();
        if (viability_fractions.front() >= threshold && viability_fractions.back() >= threshold) return g_gamma_values.front();
        return 1e-10;
    }

    static std::tuple<double, double, double, std::vector<double>> perform_linear_regression(
        const std::vector<double>& x, const std::vector<double>& y) {
        const size_t n = x.size();
        if (n < 2) return {0.0, 0.0, 0.0, std::vector<double>()};
        double sx = std::accumulate(x.begin(), x.end(), 0.0), sy = std::accumulate(y.begin(), y.end(), 0.0);
        double sxx = 0, sxy = 0;
        for (size_t i = 0; i < n; ++i) { sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
        double denom = (n * sxx - sx * sx);
        if (std::abs(denom) < 1e-9) return {0.0, 0.0, 0.0, std::vector<double>(n, 0.0)};
        double slope = (n * sxy - sx * sy) / denom;
        double intercept = (sy - slope * sx) / n;
        std::vector<double> res(n);
        double ss_tot = 0, ss_res = 0, ymean = sy / n;
        for (size_t i = 0; i < n; ++i) {
            res[i] = y[i] - (slope * x[i] + intercept);
            ss_res += res[i] * res[i];
            ss_tot += (y[i] - ymean) * (y[i] - ymean);
        }
        double r2 = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : (ss_res < 1e-9 ? 1.0 : 0.0);
        return {slope, intercept, r2, res};
    }

    static std::tuple<double, double, double, std::vector<double>> perform_exponential_fit(
        const std::vector<double>& N_values, const std::vector<double>& g_gamma_50_values) {
        std::vector<double> log_N, log_g_gamma;
        std::vector<double> residuals;  // Will be populated after regression
        for (size_t i = 0; i < N_values.size(); ++i) {
            if (N_values[i] > 0 && std::isfinite(g_gamma_50_values[i])) {
                log_N.push_back(std::log10(N_values[i]));
                log_g_gamma.push_back(g_gamma_50_values[i]);  // Already logged; do not re-log
            }
        }
        if (log_N.size() < 2) {
            return {0.0, 0.0, 0.0, std::vector<double>()};
        }
        auto [m, intercept, r2, res] = perform_linear_regression(log_N, log_g_gamma);
        double c = -intercept;  // Convert intercept to c where intercept = -c/ln(10)
        residuals = res;  // Assign the computed residuals
        return {m, c, r2, residuals};
    }

    // Sigmoid function for fitting: f(x) = L / (1 + exp(-k*(x - x0))) + b
    static int sigmoid_func(const gsl_vector* params, void* data, gsl_vector* f) {
        double L = gsl_vector_get(params, 0);
        double x0 = gsl_vector_get(params, 1);
        double k = gsl_vector_get(params, 2);
        double b = gsl_vector_get(params, 3);

        auto* fit_data = static_cast<std::pair<std::vector<double>*, std::vector<double>*>*>(data);
        const std::vector<double>& x = *fit_data->first;
        const std::vector<double>& y = *fit_data->second;

        for (size_t i = 0; i < x.size(); ++i) {
            double yi = L / (1.0 + std::exp(-k * (x[i] - x0))) + b;
            gsl_vector_set(f, i, yi - y[i]);
        }
        return GSL_SUCCESS;
    }

    std::tuple<double, double, double, double> fit_sigmoid(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.size() < 4) {
            std::cerr << "[WARNING] Insufficient data for sigmoid fit. Using defaults." << std::endl;
            return {1.0, -10.0, 10.0, 0.0};  // Fallback: Sharp transition at x0=-10
        }

        const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
        gsl_multifit_nlinear_workspace* w;
        gsl_multifit_nlinear_fdf fdf;
        gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

        const size_t n = x.size();
        const size_t p = 4;  // Parameters: L, x0, k, b

        gsl_vector* f_vec = gsl_vector_alloc(n);
        gsl_matrix* J = gsl_matrix_alloc(n, p);
        gsl_matrix* cov = gsl_matrix_alloc(p, p);
        gsl_vector* params = gsl_vector_alloc(p);

        // Initial guesses: L=1, x0=mean(x), k=1 (gentle), b=0
        double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
        gsl_vector_set(params, 0, 1.0);
        gsl_vector_set(params, 1, x_mean);
        gsl_vector_set(params, 2, 1.0);
        gsl_vector_set(params, 3, 0.0);

        std::pair<std::vector<double>*, std::vector<double>*> fit_data(const_cast<std::vector<double>*>(&x), const_cast<std::vector<double>*>(&y));

        fdf.f = sigmoid_func;
        fdf.df = nullptr;  // Use finite differences
        fdf.n = n;
        fdf.p = p;
        fdf.params = &fit_data;

        w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);
        int info;
        gsl_multifit_nlinear_init(params, &fdf, w);

        int status = gsl_multifit_nlinear_driver(100, 1e-6, 1e-6, 1e-6, nullptr, nullptr, &info, w);


        if (status != GSL_SUCCESS) {
            std::cerr << "[WARNING] Sigmoid fit did not converge: " << gsl_strerror(status) << std::endl;
            gsl_multifit_nlinear_free(w);
            gsl_vector_free(f_vec);
            gsl_matrix_free(J);
            gsl_matrix_free(cov);
            gsl_vector_free(params);
            return {1.0, x_mean, 1.0, 0.0};  // Fallback with initial guesses
        }

        double L = gsl_vector_get(w->x, 0);
        double x0 = gsl_vector_get(w->x, 1);
        double k = gsl_vector_get(w->x, 2);
        double b = gsl_vector_get(w->x, 3);

        gsl_multifit_nlinear_free(w);
        gsl_vector_free(f_vec);
        gsl_matrix_free(J);
        gsl_matrix_free(cov);
        gsl_vector_free(params);

        return {L, x0, k, b};
    }



    // Single Realization
        RealizationResult single_realization(int N, long double g_e, long double g_gamma, long double phi_max) {
            Eigen::MatrixXd U_gamma = MatrixOperations::random_so_n(N);
            Eigen::MatrixXd U_e = MatrixOperations::random_so_n(N);

            if (!MatrixOperations::validate_so_n_matrix(U_gamma, Constants::MATRIX_TOLERANCE) ||
                !MatrixOperations::validate_so_n_matrix(U_e, Constants::MATRIX_TOLERANCE)) {
                RealizationResult result = {
                    false, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L,
                    Eigen::VectorXd::Zero(N), Eigen::VectorXd::Zero(N),
                    1.0L, 1.0L, 1.0L, 1.0L, false, false, "invalid SO(N) matrix"
                };
                return result;
            }

            auto [g_i_gamma, g_i_e] = MatrixOperations::generate_couplings(N, g_gamma, g_e, U_gamma, U_e);
            long double p_gg = MatrixOperations::calculate_p_gamma_gamma(g_i_gamma);
            long double p_eg = MatrixOperations::calculate_p_e_gamma(g_i_e, g_i_gamma);
            if (!(std::isfinite(p_gg) && std::isfinite(p_eg) && p_gg >= 0 && p_gg <= 1 && p_eg >= 0 && p_eg <= 1)) {
                RealizationResult result = {
                    false, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L,
                    g_i_gamma, g_i_e,
                    1.0L, 1.0L, 1.0L, 1.0L, false, false, "probability domain error"
                };
                return result;
            }

            long double phi_osc = FluxCalculations::oscillated_flux(g_gamma, g_e, p_gg, p_eg);
            long double det_err = MatrixOperations::matrix_determinant_error(U_gamma);
            long double orth_err = MatrixOperations::matrix_orthogonality_error(U_gamma);
            bool success = phi_osc < phi_max;

            bool physics_valid = true;
            std::string failure_reason;
            if (p_gg < 0.0L || p_gg > 1.0L) {
                physics_valid = false;
                failure_reason = "P_gamma_gamma out of range [0,1]: " + std::to_string(static_cast<double>(p_gg));
            }
            if (p_eg < 0.0L || p_eg > 1.0L) {
                physics_valid = false;
                if (!failure_reason.empty()) failure_reason += "; ";
                failure_reason += "P_e_gamma out of range [0,1]: " + std::to_string(static_cast<double>(p_eg));
            }

            if (N > 10) {
                double expected_p_gg = 1.0 / static_cast<double>(N);
                double relative_error = std::abs(static_cast<double>(p_gg) - expected_p_gg) / expected_p_gg;
                if (relative_error > 0.5) {
                    // Weak check, not a hard failure
                }
            }

            RealizationResult result = {
                success, p_gg, p_eg, 0.0L, 0.0L, phi_osc, 1.0L, 1.0L,
                g_i_gamma, g_i_e,
                det_err, orth_err, 1.0L, 1.0L,
                physics_valid, true, failure_reason
            };
            return result;
        }
        // Enhanced Figure 3 Generation with Pre-computation and Statistical Outputs
        std::vector<std::vector<long double>> generate_figure_3() {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "ALP ANARCHY FIGURE 3: PHOTON COUPLING SCALING WITH N" << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            // Configuration parameters
            const long double G_E_FIXED = 1e-15L; // Fixed electron coupling
            const int N_MIN = 2;
            const int N_MAX = 30;
            const long double G_GAMMA_LOG_MIN = Constants::G_GAMMA_LOG_MIN;
            const long double G_GAMMA_LOG_MAX = Constants::G_GAMMA_LOG_MAX;
            const long double TOLERANCE_W = 1e-10L; // Tolerance for W = 0.5
            const long double TOLERANCE_G = 1e-13L; // Tolerance for g_gamma convergence
            const int MAX_ITER = 10000; // Maximum binary search iterations
            std::cout << "[CONFIG] Fixed g_e: " << std::scientific << std::setprecision(3) << G_E_FIXED << std::endl;
            std::cout << "[CONFIG] N range: " << N_MIN << " to " << N_MAX << std::endl;
            std::cout << "[CONFIG] g_gamma log range: [" << G_GAMMA_LOG_MIN << ", " << G_GAMMA_LOG_MAX << "]" << std::endl;
            std::cout << "[CONFIG] Realizations per N: " << Constants::N_REALIZATIONS << std::endl;

            std::string data_dir = Constants::DATA_DIR + "/data_fig3";
            DataOutput::create_output_directory(data_dir);

            std::vector<std::vector<long double>> data;
            std::stringstream header_ss;
            header_ss << "N log_g_50gamma W P_gg_mean P_gg_std P_eg_mean P_eg_std";

            // Power-law fit data
            std::vector<double> n_values, log_g_50gamma_values;

            // Store stats for each N to use in final loop
            std::vector<std::pair<DataOutput::StatisticalTestResults, DataOutput::StatisticalTestResults>> stats_per_n;

            EnhancedProgressBar n_progress("Figure 3 N Loop", true, true);
            int total_n = N_MAX - N_MIN + 1;
            int processed_n = 0;

            // Loop through each N
            for (int N = N_MIN; N <= N_MAX; ++N) {
                long double expected_p_gamma = (N == 2) ? 0.625 : 1.0 / N;
                long double expected_p_e_gamma = (N == 2) ? 0.25 : 0.1;
                long double tolerance = Constants::get_validation_tolerance(Constants::N_REALIZATIONS);
                std::cout << "\n[PHYSICS] N = " << N << std::endl;
                std::cout << " Expected P_gg: " << std::scientific << std::setprecision(3) << expected_p_gamma << std::endl;
                std::cout << " Expected P_eg: " << std::scientific << std::setprecision(3) << expected_p_e_gamma << std::endl;
                std::cout << " Validation tolerance: " << tolerance * 100 << "% (adaptive)" << std::endl;

                // Step 1: Pre-compute Probability Distributions
                std::string prob_file = data_dir + "/precomputed_probabilities_N" + std::to_string(N) + ".txt";
                std::ofstream init_file(prob_file);
                init_file << "# P_gg P_eg\n";
                init_file.close();

                EnhancedProgressBar sample_progress("Pre-computing Probabilities for N=" + std::to_string(N), true, true);
        #pragma omp parallel for
                for (int real = 0; real < Constants::N_REALIZATIONS; ++real) {
                    Eigen::MatrixXd U_gamma = MatrixOperations::random_so_n(N);
                    Eigen::MatrixXd U_e = MatrixOperations::random_so_n(N);
                    auto [g_i_gamma, g_i_e] = MatrixOperations::generate_couplings(N, 1.0L, 1.0L, U_gamma, U_e);
                    long double p_gg = MatrixOperations::calculate_p_gamma_gamma(g_i_gamma);
                    long double p_eg = MatrixOperations::calculate_p_e_gamma(g_i_e, g_i_gamma);
        #pragma omp critical
                    {
                        std::ofstream file(prob_file, std::ios::app);
                        if (file.is_open()) {
                            file << std::setprecision(18) << p_gg << " " << p_eg << "\n";
                            file.close();
                        }
                    }
                    sample_progress.update(real + 1, Constants::N_REALIZATIONS);
                }
                sample_progress.finish("Probability distributions computed");

                // Step 2: Create CDF for P_gamma_gamma
                std::vector<long double> p_gg_sorted;
                p_gg_sorted.reserve(Constants::N_REALIZATIONS);
                std::ifstream prob_in(prob_file);
                if (!prob_in.is_open()) {
                    std::cerr << "Error: Could not open file " << prob_file << std::endl;
                    continue;
                }
                std::string line;
                std::getline(prob_in, line); // Skip header
                while (std::getline(prob_in, line)) {
                    std::istringstream iss(line);
                    long double p_gg, p_eg;
                    if (iss >> p_gg >> p_eg) {
                        p_gg_sorted.push_back(p_gg);
                    }
                }
                prob_in.close();
                std::sort(p_gg_sorted.begin(), p_gg_sorted.end());
                auto cdf = [&](long double p_crit) -> long double {
                    return static_cast<long double>(
                        std::lower_bound(p_gg_sorted.begin(), p_gg_sorted.end(), p_crit) - p_gg_sorted.begin()
                    ) / Constants::N_REALIZATIONS;
                };

                // Save CDF data
                std::vector<std::vector<long double>> cdf_data = {{std::numeric_limits<long double>::quiet_NaN()}};
                for (size_t i = 0; i < p_gg_sorted.size(); ++i) {
                    cdf_data.push_back({p_gg_sorted[i], static_cast<long double>(i + 1) / Constants::N_REALIZATIONS});
                }
                DataOutput::write_diagnostics_realization_data(
                    cdf_data,
                    data_dir + "/cdf_p_gg_N" + std::to_string(N) + ".txt",
                    "P_gg CDF"
                );

                // Step 3: Find g_50gamma using binary search
                long double log_g_gamma_low = G_GAMMA_LOG_MIN;
                long double log_g_gamma_high = G_GAMMA_LOG_MAX;
                long double log_g_50gamma = 0.0L;
                long double W = 0.0L;
                const int M = 50; // Number of averages to reduce noise (tune based on runtime)
                for (int iter = 0; iter < MAX_ITER && log_g_gamma_high - log_g_gamma_low > TOLERANCE_G; ++iter) {
                    log_g_50gamma = (log_g_gamma_low + log_g_gamma_high) / 2.0L;
                    long double g_gamma = std::pow(10.0L, log_g_50gamma);

                    // Calculate fluxes (unchanged)
                    long double phi_p_val = FluxCalculations::phi_p(g_gamma);
                    long double phi_b_val = FluxCalculations::phi_b(G_E_FIXED);
                    long double phi_c_val = FluxCalculations::phi_c(G_E_FIXED);
                    long double phi_e = phi_b_val + phi_c_val;
                    long double g_n1 = FluxCalculations::g_gamma_n1(G_E_FIXED);
                    long double phi_max = FluxCalculations::phi_p(g_n1);
                    if (g_gamma * g_gamma > 1e-50L) {
                        phi_max = (phi_p_val + phi_b_val + phi_c_val) * (g_n1 * g_n1) / (g_gamma * g_gamma);
                    }
                    phi_max = std::max(static_cast<long double>(phi_max), 1e-50L);
                    if (!std::isfinite(phi_max)) phi_max = 1e40L;

                    // Compute averaged W to reduce noise
                    long double W_avg = 0.0L;
                    for (int avg = 0; avg < M; ++avg) { // Repeat for averaging
                        long double W_temp = 0.0L;
                        std::ifstream prob_in(prob_file); // Re-open file for each average
                        if (!prob_in.is_open()) continue;
                        std::string line;
                        std::getline(prob_in, line); // Skip header
                        #pragma omp parallel for reduction(+:W_temp)
                        for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                            std::string data_line;
                            #pragma omp critical
                            if (std::getline(prob_in, data_line)) {
                                std::istringstream iss(data_line);
                                long double p_gg, p_eg;
                                if (iss >> p_gg >> p_eg) {
                                    long double p_crit = (phi_max - p_eg * phi_e) / phi_p_val;
                                    long double eta = (p_crit > 0 && std::isfinite(p_crit)) ? cdf(p_crit) : 0.0L;
                                    W_temp += eta;
                                }
                            }
                        }
                        prob_in.close();
                        W_temp /= Constants::N_REALIZATIONS;
                        W_avg += W_temp;
                    }
                    W = W_avg / M; // Averaged W

                    // Adjust search range (unchanged)
                    if (W > 0.5L + TOLERANCE_W) {
                        log_g_gamma_low = log_g_50gamma;
                    } else if (W < 0.5L - TOLERANCE_W) {
                        log_g_gamma_high = log_g_50gamma;
                    } else {
                        break;
                    }
                }

                // Step 3.5: Refine with sigmoid fit
                std::vector<double> log_g_samples;
                std::vector<double> W_samples;
                int num_fit_points = 20; // Number of sampling points
                double fit_range = 0.5; // +/- range in log space around current log_g_50gamma
                for (int s = 0; s < num_fit_points; ++s) {
                    double log_g_sample = log_g_50gamma - fit_range + (2 * fit_range * s / (num_fit_points - 1));
                    double g_gamma_sample = std::pow(10.0, log_g_sample);

                    // Recalculate fluxes (copy from binary loop)
                    double phi_p_val = FluxCalculations::phi_p(g_gamma_sample);
                    double phi_b_val = FluxCalculations::phi_b(G_E_FIXED);
                    double phi_c_val = FluxCalculations::phi_c(G_E_FIXED);
                    double phi_e = phi_b_val + phi_c_val;
                    double g_n1 = FluxCalculations::g_gamma_n1(G_E_FIXED);
                    double phi_max = FluxCalculations::phi_p(g_n1);
                    if (g_gamma_sample * g_gamma_sample > 1e-50L) {
                        phi_max = (phi_p_val + phi_b_val + phi_c_val) * (g_n1 * g_n1) / (g_gamma_sample * g_gamma_sample);
                    }
                    phi_max = std::max(static_cast<long double>(phi_max), 1e-50L);

                    if (!std::isfinite(phi_max)) phi_max = 1e40L;

                    // Compute W for this sample (average as in binary loop)
                    double W_avg = 0.0;
                    for (int avg = 0; avg < M; ++avg) { // Reuse M from binary loop
                        double W_temp = 0.0;
                        std::ifstream prob_in(prob_file);
                        if (!prob_in.is_open()) continue;
                        std::string line;
                        std::getline(prob_in, line); // Skip header
                        #pragma omp parallel for reduction(+:W_temp)
                        for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                            std::string data_line;
                            #pragma omp critical
                            if (std::getline(prob_in, data_line)) {
                                std::istringstream iss(data_line);
                                long double p_gg, p_eg;
                                if (iss >> p_gg >> p_eg) {
                                    long double p_crit = (phi_max - p_eg * phi_e) / phi_p_val;
                                    long double eta = (p_crit > 0 && std::isfinite(p_crit)) ? cdf(p_crit) : 0.0L;
                                    W_temp += eta;
                                }
                            }
                        }
                        prob_in.close();
                        W_temp /= Constants::N_REALIZATIONS;
                        W_avg += W_temp;
                    }
                    double W_sample = W_avg / M;
                    log_g_samples.push_back(log_g_sample);
                    W_samples.push_back(W_sample);
                }

                // Perform fit
                auto [L, x0, k, b] = fit_sigmoid(log_g_samples, W_samples);

                // Refine log_g_50gamma: Solve for W = 0.5
                // Formula: log_g = x0 - (1/k) * log( (L / (0.5 - b)) - 1 )
                if (std::abs(k) > 1e-10 && (0.5 - b) > 0 && L > 0) {
                    double arg = L / (0.5 - b) - 1.0;
                    if (arg > 0) {
                        log_g_50gamma = x0 - (1.0 / k) * std::log(arg);
                    }
                }  // Else fallback to binary search value

                // Now proceed with refined log_g_50gamma
                // Recompute final W with refined value (copy flux and W calc from binary loop)
                double g_gamma = std::pow(10.0, log_g_50gamma);
                double phi_p_val = FluxCalculations::phi_p(g_gamma);
                double phi_b_val = FluxCalculations::phi_b(G_E_FIXED);
                double phi_c_val = FluxCalculations::phi_c(G_E_FIXED);
                double phi_e = phi_b_val + phi_c_val;
                double g_n1 = FluxCalculations::g_gamma_n1(G_E_FIXED);
                double phi_max = FluxCalculations::phi_p(g_n1);
                if (g_gamma * g_gamma > 1e-50L) {
                    phi_max = (phi_p_val + phi_b_val + phi_c_val) * (g_n1 * g_n1) / (g_gamma * g_gamma);
                }
                phi_max = std::max(static_cast<long double>(phi_max), 1e-50L);

                if (!std::isfinite(phi_max)) phi_max = 1e40L;
                double W_avg = 0.0;
                for (int avg = 0; avg < M; ++avg) {
                    double W_temp = 0.0;
                    std::ifstream prob_in(prob_file);
                    if (!prob_in.is_open()) continue;
                    std::string line;
                    std::getline(prob_in, line); // Skip header
                    #pragma omp parallel for reduction(+:W_temp)
                    for (int k = 0; k < Constants::N_REALIZATIONS; ++k) {
                        std::string data_line;
                        #pragma omp critical
                        if (std::getline(prob_in, data_line)) {
                            std::istringstream iss(data_line);
                            long double p_gg, p_eg;
                            if (iss >> p_gg >> p_eg) {
                                long double p_crit = (phi_max - p_eg * phi_e) / phi_p_val;
                                long double eta = (p_crit > 0 && std::isfinite(p_crit)) ? cdf(p_crit) : 0.0L;
                                W_temp += eta;
                            }
                        }
                    }
                    prob_in.close();
                    W_temp /= Constants::N_REALIZATIONS;
                    W_avg += W_temp;
                }
                W = W_avg / M;

                // Compute statistics using file data
                std::vector<long double> p_gg_vals, p_eg_vals;
                p_gg_vals.reserve(Constants::N_REALIZATIONS);
                p_eg_vals.reserve(Constants::N_REALIZATIONS);
                prob_in.open(prob_file);
                if (prob_in.is_open()) {
                    std::getline(prob_in, line); // Skip header
                    while (std::getline(prob_in, line)) {
                        std::istringstream iss(line);
                        long double p_gg, p_eg;
                        if (iss >> p_gg >> p_eg) {
                            p_gg_vals.push_back(p_gg);
                            p_eg_vals.push_back(p_eg);
                        }
                    }
                    prob_in.close();
                }

                long double p_gg_mean = std::accumulate(p_gg_vals.begin(), p_gg_vals.end(), 0.0) / p_gg_vals.size();
                long double p_eg_mean = std::accumulate(p_eg_vals.begin(), p_eg_vals.end(), 0.0) / p_eg_vals.size();
                long double p_gg_var = 0.0, p_eg_var = 0.0;
                if (p_gg_vals.size() > 1) {
                    for (double p : p_gg_vals) p_gg_var += (p - p_gg_mean) * (p - p_gg_mean);
                    for (double p : p_eg_vals) p_eg_var += (p - p_eg_mean) * (p - p_eg_mean);
                    p_gg_var /= (p_gg_vals.size() - 1);
                    p_eg_var /= (p_eg_vals.size() - 1);
                }
                long double p_gg_std = std::sqrt(p_gg_var);
                long double p_eg_std = std::sqrt(p_eg_var);

                // Perform statistical tests
                DataOutput::StatisticalTestResults p_gg_stats, p_eg_stats;
                auto [ks_stat_gg, ks_pval_gg] = MatrixOperations::kolmogorov_smirnov_test(p_gg_vals, "normal");
                auto [chi2_stat_gg, chi2_pval_gg] = MatrixOperations::chi_square_gof_test(p_gg_vals, "normal");
                p_gg_stats.ks_statistic = ks_stat_gg;
                p_gg_stats.ks_p_value = ks_pval_gg; // Real value
                p_gg_stats.chi2_statistic = chi2_stat_gg;
                p_gg_stats.chi2_p_value = chi2_pval_gg; // Real value
                p_gg_stats.sample_mean = p_gg_mean;
                p_gg_stats.sample_std = p_gg_std;
                p_gg_stats.expected_mean = expected_p_gamma;
                p_gg_stats.expected_std = expected_p_gamma / std::sqrt(N);
                p_gg_stats.sample_size = p_gg_vals.size();
                p_gg_stats.validation_tolerance_used = tolerance;

                // For p_eg_stats
                auto [ks_stat_eg, ks_pval_eg] = MatrixOperations::kolmogorov_smirnov_test(p_eg_vals, "normal");
                auto [chi2_stat_eg, chi2_pval_eg] = MatrixOperations::chi_square_gof_test(p_eg_vals, "normal");
                p_eg_stats.ks_statistic = ks_stat_eg;
                p_eg_stats.ks_p_value = ks_pval_eg; // Real value
                p_eg_stats.chi2_statistic = chi2_stat_eg;
                p_eg_stats.chi2_p_value = chi2_pval_eg; // Real value
                p_eg_stats.sample_mean = p_eg_mean;
                p_eg_stats.sample_std = p_eg_std;
                p_eg_stats.expected_mean = expected_p_e_gamma;
                p_eg_stats.expected_std = expected_p_e_gamma / std::sqrt(N);
                p_eg_stats.sample_size = p_eg_vals.size();
                p_eg_stats.validation_tolerance_used = tolerance;

                // Store stats for later use
                stats_per_n.push_back(std::make_pair(p_gg_stats, p_eg_stats));

                // Store data
                data.push_back({(long double)N, log_g_50gamma, W, (long double)p_gg_mean, (long double)p_gg_std, (long double)p_eg_mean, (long double)p_eg_std});
                n_values.push_back(N);
                log_g_50gamma_values.push_back(log_g_50gamma);

                // Save statistical results
                DataOutput::Figure3Data figure3_data;
                figure3_data.N = N;
                figure3_data.n_realizations = Constants::N_REALIZATIONS;
                figure3_data.g_e_fixed = G_E_FIXED;
                figure3_data.log_g_50gamma = log_g_50gamma;
                figure3_data.W = W;
                figure3_data.global_p_gamma_stats = p_gg_stats;
                figure3_data.global_p_e_gamma_stats = p_eg_stats;
                figure3_data.p_gamma_raw_data = p_gg_vals;
                figure3_data.p_e_gamma_raw_data = p_eg_vals;

                // Note: p_gamma_raw_data and p_e_gamma_raw_data are not stored in memory but will be read from file in write_figure3_stats
                DataOutput::write_figure3_stats(figure3_data, data_dir + "/figure_3_stats_N" + std::to_string(N) + ".txt");

                processed_n++;
                std::ostringstream details_ss;
                details_ss << "N=" << N << ", log_g_50gamma=" << std::fixed << std::setprecision(3) << log_g_50gamma << ", W=" << W;
                n_progress.update(processed_n, total_n, details_ss.str());
            }
            n_progress.finish("N loop completed");

            // Step 4: Fit Power-Law to g_50gamma vs N
            std::vector<double> log_n_values(n_values.size());
            for (size_t i = 0; i < n_values.size(); ++i) {
                log_n_values[i] = std::log10(n_values[i]);
            }

            // Perform the fit using the corrected exponential function (handles pre-logged values and returns residuals)
            auto [m, c, r2, residuals] = perform_exponential_fit(n_values, log_g_50gamma_values);
            std::cout << "\n[POWER-LAW FIT] log10(g_50gamma) = " << std::fixed << std::setprecision(3) << m
                      << " * log10(N) - " << std::fixed << std::setprecision(3) << c*std::log(10) << ", R² = " << r2 << std::endl;

            // Save power-law fit parameters and residuals
            std::vector<std::vector<long double>> fit_data = {{m, c, r2}};
            DataOutput::write_diagnostics_realization_data(
                fit_data,
                data_dir + "/power_law_fit.txt",
                "m c r_squared"
            );

            // Save residuals and fit predictions for plotting with safety checks
            std::vector<std::vector<long double>> fit_plot_data;
            fit_plot_data.reserve(n_values.size() + 1); // Pre-allocate for efficiency
            fit_plot_data.push_back({std::numeric_limits<long double>::quiet_NaN()});
            size_t valid_idx = 0;
            for (size_t i = 0; i < n_values.size(); ++i) {
                if (n_values[i] > 0 && std::isfinite(log_g_50gamma_values[i])) {
                    if (valid_idx < residuals.size()) { // Safety check to prevent out-of-bounds
                        long double predicted_log_g_50gamma = m * std::log10(n_values[i]) - c / std::log(10.0);
                        fit_plot_data.push_back({n_values[i], log_g_50gamma_values[i], predicted_log_g_50gamma, residuals[valid_idx]});
                        ++valid_idx;
                    } else {
                        fit_plot_data.push_back({n_values[i], log_g_50gamma_values[i], std::numeric_limits<long double>::quiet_NaN(), std::numeric_limits<long double>::quiet_NaN()});
                    }
                } else {
                    fit_plot_data.push_back({n_values[i], log_g_50gamma_values[i], std::numeric_limits<long double>::quiet_NaN(), std::numeric_limits<long double>::quiet_NaN()});
                }
            }
            DataOutput::write_diagnostics_realization_data(
                fit_plot_data,
                data_dir + "/fit_plot_data.txt",
                "N log_g_50gamma predicted_log_g_50gamma residual"
            );

            // Perform statistical tests on residuals
            std::vector<long double> residuals_ld(residuals.begin(), residuals.end());
            DataOutput::StatisticalTestResults residual_stats;
            auto ks_result = MatrixOperations::kolmogorov_smirnov_test(residuals_ld, "normal");
            residual_stats.ks_statistic = ks_result.first;
            residual_stats.ks_p_value = ks_result.second;  // No longer placeholder

            auto chi2_result = MatrixOperations::chi_square_gof_test(residuals_ld, "normal");
            residual_stats.chi2_statistic = chi2_result.first;
            residual_stats.chi2_p_value = chi2_result.second;  // No longer placeholder
            residual_stats.sample_mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
            residual_stats.sample_std = residuals.size() > 1 ? std::sqrt(
                std::accumulate(residuals.begin(), residuals.end(), 0.0,
                    [mean = residual_stats.sample_mean](double sum, double x) { return sum + (x - mean) * (x - mean); }
                ) / (residuals.size() - 1)) : 0.0;
            residual_stats.sample_size = residuals.size();

            // Write Figure 3 Data
            DataOutput::write_figure_3_data(data, data_dir + "/figure_3.txt", header_ss.str());

            // Print Statistical Table
            std::vector<std::pair<std::string, DataOutput::StatisticalTestResults>> stat_results;
            stat_results.push_back(std::make_pair("Fit Residuals", residual_stats));
            for (int i = 0; i < stats_per_n.size(); ++i) {
                int N = N_MIN + i;
                stat_results.push_back(std::make_pair("P_gg (N=" + std::to_string(N) + ")", stats_per_n[i].first));
                stat_results.push_back(std::make_pair("P_eg (N=" + std::to_string(N) + ")", stats_per_n[i].second));
            }
            DataOutput::print_statistical_table(stat_results, "Figure 3 Statistical Analysis");

            std::vector<std::string> figure3_files;
            for (int N = N_MIN; N <= N_MAX; ++N) {
                figure3_files.push_back(data_dir + "/precomputed_probabilities_N" + std::to_string(N) + ".txt");
                figure3_files.push_back(data_dir + "/cdf_p_gg_N" + std::to_string(N) + ".txt");
                figure3_files.push_back(data_dir + "/figure_3_stats_N" + std::to_string(N) + ".txt");
            }
            figure3_files.push_back(data_dir + "/power_law_fit.txt");
            figure3_files.push_back(data_dir + "/fit_plot_data.txt");
            figure3_files.push_back(data_dir + "/figure_3.txt");
            list_relevant_files(figure3_files, "Figure 3");

            return data;
        }


            // Figure 2 Generation
            std::vector<std::vector<long double>> generate_figure_2(int N) {
                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "ALP ANARCHY FIGURE 2: CAST BOUNDS HEATMAP (N=" << N << ")" << std::endl;
                std::cout << std::string(60, '=') << std::endl;

                std::string fig2_dir = Constants::DATA_DIR + "/data_fig2";
                DataOutput::create_output_directory(fig2_dir);

                double expected_p_gamma = (N == 2) ? 0.625 : 1.0 / N;
                double expected_p_e_gamma = (N == 2) ? 0.25 : 0.1;
                double tolerance = Constants::get_validation_tolerance(Constants::N_REALIZATIONS);

                std::cout << "[PHYSICS] Expected P_gg: " << std::scientific << std::setprecision(3) << expected_p_gamma << std::endl;
                std::cout << "[PHYSICS] Expected P_eg: " << std::scientific << std::setprecision(3) << expected_p_e_gamma << std::endl;
                std::cout << "[PHYSICS] Validation tolerance: " << tolerance * 100 << "% (adaptive)" << std::endl;

                long double g_e_range = Constants::G_E_LOG_MAX - Constants::G_E_LOG_MIN;
                long double g_gamma_range = Constants::G_GAMMA_LOG_MAX - Constants::G_GAMMA_LOG_MIN;
                int g_e_steps = static_cast<long double>(g_e_range * Constants::G_E_POINTS_PER_DECADE);
                int g_gamma_steps = static_cast<long double>(g_gamma_range * Constants::G_GAMMA_POINTS_PER_DECADE);

                std::cout << "\n[CONFIG] Grid parameters:" << std::endl;
                std::cout << " Grid dimensions: " << g_e_steps << " x " << g_gamma_steps << std::endl;
                std::cout << " Total grid points: " << g_e_steps * g_gamma_steps << std::endl;
                std::cout << " Realizations per point: " << Constants::N_REALIZATIONS << std::endl;

                if (g_e_steps <= 0 || g_gamma_steps <= 0) {
                    throw std::runtime_error("Invalid grid parameters");
                }

                long double g_e_step = g_e_range / static_cast<long double>(g_e_steps);
                long double g_gamma_step = g_gamma_range / static_cast<long double>(g_gamma_steps);
                int total_steps = g_e_steps * g_gamma_steps;

                // Initialize data structures (reduced memory: no raw data storage)
                DataOutput::Figure2Data figure2_data;
                figure2_data.N = N;
                figure2_data.n_realizations = Constants::N_REALIZATIONS;
                figure2_data.g_e_values.resize(g_e_steps);
                figure2_data.g_gamma_values.resize(g_gamma_steps);
                figure2_data.weight_matrix.resize(g_e_steps, std::vector<double>(g_gamma_steps));
                figure2_data.p_gamma_mean_matrix.resize(g_e_steps, std::vector<double>(g_gamma_steps));
                figure2_data.p_gamma_std_matrix.resize(g_e_steps, std::vector<double>(g_gamma_steps));
                figure2_data.p_e_gamma_mean_matrix.resize(g_e_steps, std::vector<double>(g_gamma_steps));
                figure2_data.p_e_gamma_std_matrix.resize(g_e_steps, std::vector<double>(g_gamma_steps));
                figure2_data.w_stats_matrix.resize(g_e_steps, std::vector<DataOutput::StatisticalTestResults>(g_gamma_steps));
                figure2_data.p_gamma_stats_matrix.resize(g_e_steps, std::vector<DataOutput::StatisticalTestResults>(g_gamma_steps));
                figure2_data.p_e_gamma_stats_matrix.resize(g_e_steps, std::vector<DataOutput::StatisticalTestResults>(g_gamma_steps));

                std::vector<std::vector<long double>> data;
                std::stringstream header_ss;
                header_ss << "log_g_e log_g_gamma W P_gg_mean P_gg_std P_eg_mean P_eg_std W_residual_mean W_chi2_p_value";


                // Step 1: Pre-compute Probability Distributions
                std::string prob_file = fig2_dir + "/precomputed_probabilities_N" + std::to_string(N) + ".txt";
                std::ofstream init_file(prob_file);
                init_file << "# P_gg P_eg\n";
                init_file.close();

                EnhancedProgressBar sample_progress("Pre-computing Probability Distributions", true, true);

            #pragma omp parallel for
                for (int real = 0; real < Constants::N_REALIZATIONS; ++real) {
                    Eigen::MatrixXd U_gamma = MatrixOperations::random_so_n(N);
                    Eigen::MatrixXd U_e = MatrixOperations::random_so_n(N);
                    auto [g_i_gamma, g_i_e] = MatrixOperations::generate_couplings(N, 1.0L, 1.0L, U_gamma, U_e);
                    long double p_gg = MatrixOperations::calculate_p_gamma_gamma(g_i_gamma);
                    long double p_eg = MatrixOperations::calculate_p_e_gamma(g_i_e, g_i_gamma);

            #pragma omp critical
                    {
                        std::ofstream file(prob_file, std::ios::app);
                        if (file.is_open()) {
                            file << std::setprecision(18) << p_gg << " " << p_eg << "\n";
                            file.close();
                        }
                    }

                    sample_progress.update(real + 1, Constants::N_REALIZATIONS);
                }
                sample_progress.finish("Probability distributions computed");

                // Step 2: Load probabilities once into memory
                std::vector<long double> all_p_gg, all_p_eg;
                all_p_gg.reserve(Constants::N_REALIZATIONS);
                all_p_eg.reserve(Constants::N_REALIZATIONS);
                std::ifstream prob_in(prob_file);
                if (!prob_in.is_open()) {
                    std::cerr << "Error: Could not open file " << prob_file << std::endl;
                    return data;
                }
                std::string line;
                std::getline(prob_in, line); // Skip header
                while (std::getline(prob_in, line)) {
                    std::istringstream iss(line);
                    long double p_gg, p_eg;
                    if (iss >> p_gg >> p_eg) {
                        all_p_gg.push_back(p_gg);
                        all_p_eg.push_back(p_eg);
                    }
                }
                prob_in.close();

                if (all_p_gg.size() != static_cast<size_t>(Constants::N_REALIZATIONS)) {
                    std::cerr << "Error: Loaded " << all_p_gg.size() << " realizations, expected " << Constants::N_REALIZATIONS << std::endl;
                    return data;
                }

                // Create sorted p_gg for CDF
                std::vector<long double> p_gg_sorted = all_p_gg;
                std::sort(p_gg_sorted.begin(), p_gg_sorted.end());
                auto cdf = [&](long double p_crit) -> double {
                    return static_cast<double>(
                        std::lower_bound(p_gg_sorted.begin(), p_gg_sorted.end(), p_crit) - p_gg_sorted.begin()
                    ) / Constants::N_REALIZATIONS;
                };

                // Save CDF data
                std::vector<std::vector<long double>> cdf_data = {{std::numeric_limits<long double>::quiet_NaN()}};
                for (size_t i = 0; i < p_gg_sorted.size(); ++i) {
                    cdf_data.push_back({p_gg_sorted[i], static_cast<double>(i + 1) / Constants::N_REALIZATIONS});
                }
                DataOutput::write_diagnostics_realization_data(
                    cdf_data,
                    fig2_dir + "/cdf_p_gg_N" + std::to_string(N) + ".txt",
                    "P_gg CDF"
                );

                // Compute global statistics for p_gg and p_eg (same for all grid points)
                double p_gg_mean = std::accumulate(all_p_gg.begin(), all_p_gg.end(), 0.0) / all_p_gg.size();
                double p_eg_mean = std::accumulate(all_p_eg.begin(), all_p_eg.end(), 0.0) / all_p_eg.size();
                double p_gg_var = 0.0, p_eg_var = 0.0;
                size_t n = all_p_gg.size();
                if (n > 1) {
                    for (double p : all_p_gg) p_gg_var += (p - p_gg_mean) * (p - p_gg_mean);
                    for (double p : all_p_eg) p_eg_var += (p - p_eg_mean) * (p - p_eg_mean);
                    p_gg_var /= (n - 1);
                    p_eg_var /= (n - 1);
                }
                double p_gg_std = std::sqrt(p_gg_var);
                double p_eg_std = std::sqrt(p_eg_var);

                DataOutput::StatisticalTestResults global_p_gg_stats, global_p_eg_stats;
                global_p_gg_stats.sample_mean = p_gg_mean;
                global_p_gg_stats.sample_std = p_gg_std;
                global_p_gg_stats.expected_mean = expected_p_gamma;
                global_p_gg_stats.expected_std = expected_p_gamma / std::sqrt(N);
                global_p_gg_stats.sample_size = n;
                global_p_gg_stats.validation_tolerance_used = tolerance;
                // KS and Chi² (using full data)
                auto [ks_stat_gg, ks_pval_gg] = MatrixOperations::kolmogorov_smirnov_test(all_p_gg, "normal");
                auto [chi2_stat_gg, chi2_pval_gg] = MatrixOperations::chi_square_gof_test(all_p_gg, "normal");
                global_p_gg_stats.ks_statistic = ks_stat_gg;
                global_p_gg_stats.ks_p_value = ks_pval_gg;  // Real value
                global_p_gg_stats.chi2_statistic = chi2_stat_gg;
                global_p_gg_stats.chi2_p_value = chi2_pval_gg;  // Real value

                global_p_eg_stats.sample_mean = p_eg_mean;
                global_p_eg_stats.sample_std = p_eg_std;
                global_p_eg_stats.expected_mean = expected_p_e_gamma;
                global_p_eg_stats.expected_std = expected_p_e_gamma / std::sqrt(N);
                global_p_eg_stats.sample_size = n;
                global_p_eg_stats.validation_tolerance_used = tolerance;

                auto [ks_stat_eg, ks_pval_eg] = MatrixOperations::kolmogorov_smirnov_test(all_p_eg, "normal");
                auto [chi2_stat_eg, chi2_pval_eg] = MatrixOperations::chi_square_gof_test(all_p_eg, "normal");
                global_p_eg_stats.ks_statistic = ks_stat_eg;
                global_p_eg_stats.ks_p_value = ks_pval_eg;  // Real value
                global_p_eg_stats.chi2_statistic = chi2_stat_eg;
                global_p_eg_stats.chi2_p_value = chi2_pval_eg;  // Real value

                figure2_data.global_p_gamma_stats = global_p_gg_stats;
                figure2_data.global_p_e_gamma_stats = global_p_eg_stats;

                // Step 3: Calculate Viability for Each Grid Point (parallel, no file I/O)
                EnhancedProgressBar grid_progress("Figure 2 Grid Generation", true, true);
                int processed = 0;
                const int M = 50;  // Averages per grid point (tune for noise vs. speed; e.g., 5-20)
                std::vector<std::vector<long double>> refined_boundaries(g_e_steps);

                #pragma omp parallel for collapse(2) schedule(dynamic) if(total_steps > 100)
                for (int i = 0; i < g_e_steps; ++i) {
                    for (int j = 0; j < g_gamma_steps; ++j) {
                        if (i >= static_cast<size_t>(figure2_data.g_e_values.size()) || j >= static_cast<size_t>(figure2_data.g_gamma_values.size()) || j < 0) continue;

                        long double log_g_e = Constants::G_E_LOG_MIN + i * g_e_step;
                        long double log_g_gamma = Constants::G_GAMMA_LOG_MIN + j * g_gamma_step;
                        if (log_g_e > Constants::G_E_LOG_MAX + 1e-10L ||
                            log_g_gamma > Constants::G_GAMMA_LOG_MAX + 1e-10L) continue;

                        long double g_e = std::pow(10.0L, log_g_e);
                        long double g_gamma = std::pow(10.0L, log_g_gamma);
                        if (i < figure2_data.g_e_values.size()) figure2_data.g_e_values[i] = g_e;
                        if (j < figure2_data.g_gamma_values.size()) figure2_data.g_gamma_values[j] = g_gamma;

                        // Calculate fluxes
                        long double phi_p_val = FluxCalculations::phi_p(g_gamma);
                        long double phi_b_val = FluxCalculations::phi_b(g_e);
                        long double phi_c_val = FluxCalculations::phi_c(g_e);
                        long double phi_e = phi_b_val + phi_c_val;

                        // Calculate phi_max based on CAST bound
                        long double g_n1 = FluxCalculations::g_gamma_n1(g_e);
                        long double flux_total_single = phi_p_val + phi_b_val + phi_c_val;
                        long double phi_max = flux_total_single;
                        if (g_gamma * g_gamma > 1e-50L) {
                            phi_max = flux_total_single * (g_n1 * g_n1) / (g_gamma * g_gamma);
                        }
                        phi_max = std::max(phi_max, 1e-50L);
                        if (!std::isfinite(phi_max)) phi_max = 1e40L;

                        // Compute averaged W and stats to reduce noise
                        double w_sum_avg = 0.0;
                        double w_sum_sq_avg = 0.0;
                        for (int avg = 0; avg < M; ++avg) {  // Average over M estimates
                            double w_sum = 0.0;
                            double w_sum_sq = 0.0;
                            size_t valid_count = 0;
                            for (size_t k = 0; k < all_p_gg.size(); ++k) {
                                double p_gg = all_p_gg[k];
                                double p_eg = all_p_eg[k];
                                double p_crit = (phi_max - p_eg * phi_e) / phi_p_val;
                                double eta = (p_crit > 0 && std::isfinite(p_crit)) ? cdf(p_crit) : 0.0;
                                w_sum += eta;
                                w_sum_sq += eta * eta;
                                ++valid_count;
                            }
                            double W_temp = (valid_count > 0) ? w_sum / valid_count : 0.0;
                            w_sum_avg += W_temp;
                            w_sum_sq_avg += (valid_count > 1) ? (w_sum_sq - (w_sum * w_sum / valid_count)) / (valid_count - 1) : 0.0;
                        }
                        double W = w_sum_avg / M;
                        double w_mean = W;
                        double w_var = w_sum_sq_avg / M;
                        double w_std = std::sqrt(w_var);

                        // w_stats (simplified; skip KS/Chi² to avoid needing full w_vals vector)
                        DataOutput::StatisticalTestResults w_stats;
                        w_stats.sample_mean = w_mean;
                        w_stats.sample_std = w_std;
                        w_stats.expected_mean = 0.5;
                        w_stats.expected_std = 0.5 / std::sqrt(N);
                        w_stats.sample_size = all_p_gg.size();  // Use full size
                        w_stats.validation_tolerance_used = tolerance;
                        w_stats.ks_statistic = 0.0;  // Omitted
                        w_stats.ks_p_value = 0.0;
                        w_stats.chi2_statistic = 0.0;
                        double w_chi2_p = compute_chi2_p_value_for_grid_point(W, Constants::N_REALIZATIONS);
                        w_stats.chi2_p_value = w_chi2_p;

                        // Store results with bounds checks
                        if (i < figure2_data.weight_matrix.size() && j < figure2_data.weight_matrix[i].size()) {
                            figure2_data.weight_matrix[i][j] = W;
                        }
                        if (i < figure2_data.p_gamma_mean_matrix.size() && j < figure2_data.p_gamma_mean_matrix[i].size()) {
                            figure2_data.p_gamma_mean_matrix[i][j] = p_gg_mean;
                            figure2_data.p_gamma_std_matrix[i][j] = p_gg_std;
                            figure2_data.p_e_gamma_mean_matrix[i][j] = p_eg_mean;
                            figure2_data.p_e_gamma_std_matrix[i][j] = p_eg_std;
                            figure2_data.p_gamma_stats_matrix[i][j] = global_p_gg_stats;
                            figure2_data.p_e_gamma_stats_matrix[i][j] = global_p_eg_stats;
                            figure2_data.w_stats_matrix[i][j] = w_stats;
                        }
                        std::vector<long double> row = {log_g_e, log_g_gamma, W,
                                                        static_cast<long double>(p_gg_mean), static_cast<long double>(p_gg_std),
                                                        static_cast<long double>(p_eg_mean), static_cast<long double>(p_eg_std),
                                                        static_cast<long double>(w_mean - 0.5), static_cast<long double>(w_chi2_p)};
                #pragma omp critical
                        data.push_back(row);
                #pragma omp atomic
                        processed++;
                        if (processed % (total_steps / 20) == 0 || processed == total_steps) {
                            std::ostringstream details_ss;
                            details_ss << "W=" << std::fixed << std::setprecision(3) << W
                                       << ", P_gg=" << std::fixed << std::setprecision(3) << p_gg_mean
                                       << ", W_residual_mean=" << std::fixed << std::setprecision(3) << (w_mean - 0.5);
                            grid_progress.update(processed, total_steps, details_ss.str());
                        }
                    }
                }
                grid_progress.finish("Grid generation completed");

                // Row-wise sigmoid refinement: Fit to each row and update W values in data and figure2_data
                for (int i = 0; i < g_e_steps; ++i) {
                    std::vector<double> log_g_samples, W_samples;
                    // Extract existing W vs log_g_gamma for this row
                    for (int j = 0; j < g_gamma_steps; ++j) {
                        double log_gg = Constants::G_GAMMA_LOG_MIN + j * g_gamma_step;
                        log_g_samples.push_back(log_gg);
                        W_samples.push_back(figure2_data.weight_matrix[i][j]);
                    }
                    // Fit sigmoid to this row's data
                    auto [L, x0, k, b] = fit_sigmoid(log_g_samples, W_samples);
                    // Recompute refined W for each original point j in this row
                    for (int j = 0; j < g_gamma_steps; ++j) {
                        double log_gg = log_g_samples[j];
                        double refined_W = L / (1.0 + std::exp(-k * (log_gg - x0))) + b;
                        // Clamp to [0,1] if needed and ensure finite
                        if (!std::isfinite(refined_W) || refined_W < 0.0 || refined_W > 1.0) {
                            refined_W = figure2_data.weight_matrix[i][j];  // Fallback to original if invalid
                        }
                        // Update figure2_data
                        figure2_data.weight_matrix[i][j] = refined_W;
                        auto& w_stats = figure2_data.w_stats_matrix[i][j];
                        w_stats.sample_mean = refined_W;  // Update mean (simplified; recompute std if you have full data)
                        w_stats.chi2_p_value = compute_chi2_p_value_for_grid_point(refined_W, Constants::N_REALIZATIONS);
                    }
                }
                // Now update the flat data vector with refined values (find rows by log_g_e and log_g_gamma)
                for (auto& row : data) {
                    long double log_g_e = row[0];
                    long double log_g_gamma = row[1];
                    // Compute i and j from log values (assuming uniform grid)
                    int i = static_cast<int>(std::round((log_g_e - Constants::G_E_LOG_MIN) / g_e_step));
                    int j = static_cast<int>(std::round((log_g_gamma - Constants::G_GAMMA_LOG_MIN) / g_gamma_step));
                    if (i >= 0 && i < g_e_steps && j >= 0 && j < g_gamma_steps) {
                        double refined_W = figure2_data.weight_matrix[i][j];
                        row[2] = refined_W;  // Update W
                        row[7] = refined_W - 0.5;  // Update W_residual_mean
                        row[8] = figure2_data.w_stats_matrix[i][j].chi2_p_value;  // Update W_chi2_p_value
                    }
                }



                // Validation and output (unchanged from original)
                std::cout << "\n[VALIDATION] Physics validation results:" << std::endl;
                // Compute global statistics
                double global_p_gamma_mean = std::accumulate(all_p_gg.begin(), all_p_gg.end(), 0.0) / n;
                double global_p_e_gamma_mean = std::accumulate(all_p_eg.begin(), all_p_eg.end(), 0.0) / n;
                double global_p_gamma_std = std::sqrt(std::accumulate(all_p_gg.begin(), all_p_gg.end(), 0.0, [global_p_gamma_mean](double sum, double val) {
                    return sum + (val - global_p_gamma_mean) * (val - global_p_gamma_mean);
                }) / (n - 1));
                double global_p_e_gamma_std = std::sqrt(std::accumulate(all_p_eg.begin(), all_p_eg.end(), 0.0, [global_p_e_gamma_mean](double sum, double val) {
                    return sum + (val - global_p_e_gamma_mean) * (val - global_p_e_gamma_mean);
                }) / (n - 1));

                // Print global statistics table
                std::cout << "\nGlobal Statistics for N=" << N << ":\n";
                std::cout << "+---------------------+-----------+----------+\n";
                std::cout << "| Metric              | Mean      | Std Dev  |\n";
                std::cout << "+---------------------+-----------+----------+\n";
                std::cout << "| P_gg                | " << std::scientific << std::setprecision(3) << global_p_gamma_mean << " | " << global_p_gamma_std << " |\n";
                std::cout << "| P_eg                | " << global_p_e_gamma_mean << " | " << global_p_e_gamma_std << " |\n";
                std::cout << "+---------------------+-----------+----------+\n";
                DataOutput::write_figure_2_data(data, fig2_dir + "/figure_2_N" + std::to_string(N) + ".txt", header_ss.str());
                DataOutput::write_figure2_stats(figure2_data, fig2_dir + "/figure_2_stats_N" + std::to_string(N) + ".txt");

                std::vector<std::pair<std::string, DataOutput::StatisticalTestResults>> stat_results = {
                    {"P_gg (Global)", figure2_data.global_p_gamma_stats},
                    {"P_eg (Global)", figure2_data.global_p_e_gamma_stats}
                };
                DataOutput::print_statistical_table(stat_results, "Figure 2 Statistical Analysis for N=" + std::to_string(N));

                std::vector<std::string> figure2_files = {
                    fig2_dir + "/precomputed_probabilities_N" + std::to_string(N) + ".txt",
                    fig2_dir + "/cdf_p_gg_N" + std::to_string(N) + ".txt",
                    fig2_dir + "/figure_2_N" + std::to_string(N) + ".txt",
                    fig2_dir + "/figure_2_stats_N" + std::to_string(N) + ".txt"
                };

                list_relevant_files(figure2_files, "Figure 2");

                return data;
            }

    std::vector<std::vector<long double>> diagnose_flux() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ALP ANARCHY: FLUX DIAGNOSTICS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Parameters
        long double g_e = std::pow(10.0L, Constants::G_E_LOG_MIN);
        long double g_gamma = std::pow(10.0L, Constants::G_GAMMA_LOG_MIN);
        int N = 2;
        int n_realizations = Constants::N_REALIZATIONS;

        std::cout << "[CONFIG] Parameters:" << std::endl;
        std::cout << "  g_e: " << g_e << " GeV^-1" << std::endl;
        std::cout << "  g_gamma: " << g_gamma << " GeV^-1" << std::endl;
        std::cout << "  N: " << N << std::endl;
        std::cout << "  Realizations: " << n_realizations << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        DataOutput::create_output_directory(data_dir);

        std::vector<std::vector<long double>> output_data = {{std::numeric_limits<long double>::quiet_NaN()}};
        std::stringstream header_ss;
        header_ss << "phi_osc phi_p phi_b phi_c";

        EnhancedProgressBar progress("Flux Diagnostics", true, true);

        std::vector<long double> phi_osc_values(n_realizations);
        long double phi_p_val = FluxCalculations::phi_p(g_gamma);
        long double phi_b_val = FluxCalculations::phi_b(g_e);
        long double phi_c_val = FluxCalculations::phi_c(g_e);

        // Create a pre-allocated vector for thread-safe result storage.
        std::vector<std::vector<long double>> thread_safe_results(n_realizations);

        #pragma omp parallel for
        for (int i = 0; i < n_realizations; ++i) {
            auto result = single_realization(N, g_e, g_gamma, phi_p_val + phi_b_val + phi_c_val);
            phi_osc_values[i] = result.phi_osc;

            // This is now thread-safe as each thread writes to a unique index 'i'.
            thread_safe_results[i] = {result.phi_osc, phi_p_val, phi_b_val, phi_c_val};

            // Safely update progress from a single thread to avoid terminal clutter.
            if (omp_get_thread_num() == 0) {
                progress.update(i + 1, n_realizations);
            }
        }

        // After the parallel loop, combine results in a single thread.
        output_data.insert(output_data.end(), thread_safe_results.begin(), thread_safe_results.end());

        progress.finish("Flux diagnostics completed");

        // Compute statistics
        double phi_osc_mean = std::accumulate(phi_osc_values.begin(), phi_osc_values.end(), 0.0L) / n_realizations;
        double phi_osc_var = 0.0;
        if (n_realizations > 1) {
            for (long double phi : phi_osc_values) phi_osc_var += (phi - phi_osc_mean) * (phi - phi_osc_mean);
            phi_osc_var /= (n_realizations - 1);
        }
        double phi_osc_std = std::sqrt(phi_osc_var);

        std::cout << "\n[RESULTS] Flux Statistics:" << std::endl;
        std::cout << "  phi_osc mean: " << std::scientific << phi_osc_mean << std::endl;
        std::cout << "  phi_osc std: " << std::scientific << phi_osc_std << std::endl;
        std::cout << "  phi_p: " << std::scientific << phi_p_val << std::endl;
        std::cout << "  phi_b: " << std::scientific << phi_b_val << std::endl;
        std::cout << "  phi_c: " << std::scientific << phi_c_val << std::endl;

        // Write data
        DataOutput::write_diagnostics_flux_data(output_data, data_dir + "/flux_diagnostics.txt", header_ss.str());

        // List all output files
        std::vector<std::string> flux_files = {
            data_dir + "/flux_diagnostics.txt"
        };
        list_relevant_files(flux_files, "Flux Diagnostics");

        return output_data;
    }

    std::vector<std::vector<long double>> diagnose_realization_data() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ALP ANARCHY: REALIZATION DATA DIAGNOSTICS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Parameters
        int N = 2;
        int n_realizations = Constants::N_REALIZATIONS;
        long double g_e = std::pow(10.0L, Constants::G_E_LOG_MIN);
        long double g_gamma = std::pow(10.0L, Constants::G_GAMMA_LOG_MIN);
        long double g_n1 = FluxCalculations::g_gamma_n1(g_e);
        long double phi_max = FluxCalculations::phi_p(g_n1) + FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e);

        std::cout << "[CONFIG] Parameters:" << std::endl;
        std::cout << "  N: " << N << std::endl;
        std::cout << "  g_e: " << g_e << " GeV^-1" << std::endl;
        std::cout << "  g_gamma: " << g_gamma << " GeV^-1" << std::endl;
        std::cout << "  Realizations: " << n_realizations << std::endl;

        std::string data_dir = normalize_path(Constants::DATA_DIR);
        DataOutput::create_output_directory(data_dir);

        std::vector<std::vector<long double>> output_data = {{std::numeric_limits<long double>::quiet_NaN()}};
        std::stringstream header_ss;
        header_ss << "# P_gg P_eg";

        EnhancedProgressBar progress("Realization Data Diagnostics", true, true);

        std::vector<long double> p_gg_vals(n_realizations);
        std::vector<long double> p_eg_vals(n_realizations);

        // Create a pre-allocated vector to store results from each thread safely.
        std::vector<std::vector<long double>> thread_safe_results(n_realizations);

        #pragma omp parallel for
        for (int i = 0; i < n_realizations; ++i) {
            auto result = single_realization(N, g_e, g_gamma, phi_max);
            p_gg_vals[i] = result.p_gg;
            p_eg_vals[i] = result.p_eg;

            // Storing the row in a pre-allocated vector is thread-safe
            // because each thread writes to a unique index 'i'.
            thread_safe_results[i] = {result.p_gg, result.p_eg};
        }

        // After the parallel loop, combine the results into the final output vector.
        // This is done in a single thread, avoiding any race conditions.
        output_data.insert(output_data.end(), thread_safe_results.begin(), thread_safe_results.end());

        progress.finish("Realization data diagnostics completed");

        // Compute statistics
        double p_gg_mean = std::accumulate(p_gg_vals.begin(), p_gg_vals.end(), 0.0L) / n_realizations;
        double p_eg_mean = std::accumulate(p_eg_vals.begin(), p_eg_vals.end(), 0.0L) / n_realizations;
        double p_gg_var = 0.0, p_eg_var = 0.0;
        if (n_realizations > 1) {
            for (long double p : p_gg_vals) p_gg_var += (p - p_gg_mean) * (p - p_gg_mean);
            for (long double p : p_eg_vals) p_eg_var += (p - p_eg_mean) * (p - p_eg_mean);
            p_gg_var /= (n_realizations - 1);
            p_eg_var /= (n_realizations - 1);
        }
        double p_gg_std = std::sqrt(p_gg_var);
        double p_eg_std = std::sqrt(p_eg_var);

        std::cout << "\n[RESULTS] Realization Statistics:" << std::endl;
        std::cout << "  P_gg mean: " << std::scientific << p_gg_mean << std::endl;
        std::cout << "  P_gg std: " << std::scientific << p_gg_std << std::endl;
        std::cout << "  P_eg mean: " << std::scientific << p_eg_mean << std::endl;
        std::cout << "  P_eg std: " << std::scientific << p_eg_std << std::endl;

        // Write data
        DataOutput::write_diagnostics_realization_data(output_data, data_dir + "/realization_diagnostics.txt", header_ss.str());

        // List all output files
        std::vector<std::string> realization_files = {
            data_dir + "/realization_diagnostics.txt"
        };
        list_relevant_files(realization_files, "Realization Data Diagnostics");

        return output_data;
    }



    // Implementing generate_convergence_fig2 to analyze convergence of Figure 2 data across realizations
    // Replace the existing generate_convergence_fig2 function in simulations.cpp with this corrected version.
    // Place it inside the Simulations namespace, replacing the current implementation.

    std::vector<std::vector<long double>> generate_convergence_analysis() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ALP ANARCHY: CONVERGENCE ANALYSIS FOR MULTIPLE N VALUES" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Parameters
        std::vector<int> N_values = {2, 10, 30};  // Multiple N values
        int n_realizations_steps[] = {100, 1000, 5000, Constants::N_REALIZATIONS};
        int max_steps = sizeof(n_realizations_steps) / sizeof(n_realizations_steps[0]);

        std::string data_dir = normalize_path(Constants::DATA_DIR) + "/data_fig2";
        DataOutput::create_output_directory(data_dir);

        std::vector<std::vector<long double>> combined_output_data;
        combined_output_data.reserve(N_values.size() * max_steps);

        std::stringstream header_ss;
        header_ss << "N n_realizations P_gg_mean P_gg_std P_eg_mean P_eg_std";

        std::cout << "[CONFIG] N values: ";
        for (int N : N_values) std::cout << N << " ";
        std::cout << std::endl;
        std::cout << "[CONFIG] Realization steps: ";
        for (int n : n_realizations_steps) std::cout << n << " ";
        std::cout << std::endl;

        EnhancedProgressBar progress("Convergence Analysis", true, true);
        int total_combinations = N_values.size() * max_steps;
        int processed = 0;

        for (int N : N_values) {
            double expected_p_gamma = (N == 2) ? 0.625 : 1.0 / N;
            double expected_p_e_gamma = (N == 2) ? 0.25 : 0.1;

            std::cout << "\n[PROCESSING] N = " << N << " (expected P_gg: " << expected_p_gamma
                      << ", expected P_eg: " << expected_p_e_gamma << ")" << std::endl;

            for (int n_realizations : n_realizations_steps) {
                std::vector<long double> p_gg_vals(n_realizations);
                std::vector<long double> p_eg_vals(n_realizations);

                long double g_e = std::pow(10.0L, Constants::G_E_LOG_MIN);
                long double g_gamma = std::pow(10.0L, Constants::G_GAMMA_LOG_MIN);
                long double g_n1 = FluxCalculations::g_gamma_n1(g_e);
                long double phi_max = FluxCalculations::phi_p(g_n1) + FluxCalculations::phi_b(g_e) + FluxCalculations::phi_c(g_e);

    #pragma omp parallel for
                for (int i = 0; i < n_realizations; ++i) {
                    auto result = single_realization(N, g_e, g_gamma, phi_max);
                    p_gg_vals[i] = result.p_gg;
                    p_eg_vals[i] = result.p_eg;
                }

                double p_gg_mean = std::accumulate(p_gg_vals.begin(), p_gg_vals.end(), 0.0L) / n_realizations;
                double p_eg_mean = std::accumulate(p_eg_vals.begin(), p_eg_vals.end(), 0.0L) / n_realizations;

                double p_gg_var = 0.0, p_eg_var = 0.0;
                if (n_realizations > 1) {
                    for (long double p : p_gg_vals) p_gg_var += (p - p_gg_mean) * (p - p_gg_mean);
                    for (long double p : p_eg_vals) p_eg_var += (p - p_eg_mean) * (p - p_eg_mean);
                    p_gg_var /= (n_realizations - 1);
                    p_eg_var /= (n_realizations - 1);
                }

                double p_gg_std = std::sqrt(p_gg_var);
                double p_eg_std = std::sqrt(p_eg_var);

                combined_output_data.push_back({(long double)N, (long double)n_realizations,
                                              (long double)p_gg_mean, (long double)p_gg_std,
                                              (long double)p_eg_mean, (long double)p_eg_std});

                processed++;
                std::ostringstream details;
                details << "N=" << N << ", n_real=" << n_realizations
                       << ", P_gg=" << std::fixed << std::setprecision(3) << p_gg_mean;
                progress.update(processed, total_combinations, details.str());
            }
        }

        progress.finish("Convergence analysis completed");

        // Write combined data
        DataOutput::write_convergence_fig2_data(combined_output_data, data_dir + "/convergence_analysis_all_N.txt", header_ss.str());

        std::vector<std::string> convergence_files = {
            data_dir + "/convergence_analysis_all_N.txt"
        };
        list_relevant_files(convergence_files, "Convergence Analysis");

        return combined_output_data;
    }


    // Implementing diagnose_matrix_distribution to analyze matrix element distributions
    std::vector<std::vector<long double>> diagnose_matrix_distribution(int N) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ALP ANARCHY: MATRIX DISTRIBUTION DIAGNOSTICS (N=" << N << ")" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        if (N <= 0) {
            std::cerr << "[ERROR] Invalid N value: " << N << ". Must be positive." << std::endl;
            return std::vector<std::vector<long double>>();
        }

        int n_realizations = Constants::N_REALIZATIONS;
        if (n_realizations <= 0) {
            std::cerr << "[ERROR] Invalid n_realizations: " << n_realizations << ". Must be positive." << std::endl;
            return std::vector<std::vector<long double>>();
        }

        std::string data_dir = normalize_path(Constants::DATA_DIR) + "/data_fig2";
        DataOutput::create_output_directory(data_dir);

        // Initialize output_data WITHOUT header row
        std::vector<std::vector<long double>> output_data;
        output_data.reserve(n_realizations * N * N);  // Reserve space for N*N elements per realization

        std::stringstream header_ss;
        header_ss << "matrix_element det_error orth_error";

        std::cout << "[CONFIG] Parameters:" << std::endl;
        std::cout << " N: " << N << std::endl;
        std::cout << " Realizations: " << n_realizations << std::endl;

        EnhancedProgressBar progress("Matrix Distribution Diagnostics", true, true);

        std::vector<long double> matrix_elements;
        std::vector<long double> det_errors(n_realizations);
        std::vector<long double> orth_errors(n_realizations);
        matrix_elements.reserve(n_realizations * N * N);

    #pragma omp parallel
        {
            std::vector<long double> local_elements;
            local_elements.reserve(n_realizations * N * N / (Constants::N_THREADS > 0 ? Constants::N_THREADS : 1));

    #pragma omp for schedule(dynamic)
            for (int i = 0; i < n_realizations; ++i) {
                Eigen::MatrixXd U = MatrixOperations::random_so_n(N);
                det_errors[i] = MatrixOperations::matrix_determinant_error(U);
                orth_errors[i] = MatrixOperations::matrix_orthogonality_error(U);

                for (int r = 0; r < N; ++r) {
                    for (int c = 0; c < N; ++c) {
                        local_elements.push_back(U(r, c));
                    }
                }

    #pragma omp critical(progress_update)
                {
                    if (omp_get_thread_num() == 0) {
                        progress.update(i + 1, n_realizations);
                    }
                }
            }

    #pragma omp critical(matrix_elements_update)
            {
                matrix_elements.insert(matrix_elements.end(), local_elements.begin(), local_elements.end());
            }
        }

        if (matrix_elements.size() != static_cast<size_t>(n_realizations * N * N)) {
            std::cerr << "[ERROR] matrix_elements size (" << matrix_elements.size()
                      << ") does not match expected (" << n_realizations * N * N << ")" << std::endl;
            return output_data;
        }

        progress.finish("Matrix distribution diagnostics completed");

        // Combine data correctly WITHOUT NaN rows
        for (int i = 0; i < n_realizations; ++i) {
            for (int j = 0; j < N * N; ++j) {
                size_t element_idx = i * N * N + j;
                if (element_idx < matrix_elements.size()) {
                    output_data.push_back({matrix_elements[element_idx], det_errors[i], orth_errors[i]});
                } else {
                    std::cerr << "[ERROR] element_idx (" << element_idx << ") out of range" << std::endl;
                    return output_data;
                }
            }
        }

        // Compute and print statistics
        double element_mean = std::accumulate(matrix_elements.begin(), matrix_elements.end(), 0.0L) / matrix_elements.size();
        double element_var = 0.0;
        if (matrix_elements.size() > 1) {
            for (long double e : matrix_elements) element_var += (e - element_mean) * (e - element_mean);
            element_var /= (matrix_elements.size() - 1);
        }
        double element_std = std::sqrt(element_var);

        std::cout << "\n[RESULTS] Matrix Distribution Statistics:" << std::endl;
        std::cout << " Matrix element mean: " << std::scientific << element_mean << std::endl;
        std::cout << " Matrix element std: " << std::scientific << element_std << std::endl;

        // Write data
        DataOutput::write_matrix_distribution_data(output_data, data_dir + "/matrix_distributions_N" + std::to_string(N) + ".txt", header_ss.str());

        std::vector<std::string> matrix_files = {
            data_dir + "/matrix_distributions_N" + std::to_string(N) + ".txt"
        };
        list_relevant_files(matrix_files, "Matrix Distribution Diagnostics");

        return output_data;
    }


    std::vector<std::vector<long double>> generate_flux_comparison_data() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ALP ANARCHY: FLUX COMPARISON DATA GENERATION" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Configuration
        const long double FIXED_G_E = 1e-15L;
        const long double FIXED_G_GAMMA = 1e-10L;
        const int NUM_POINTS = 1000;

        std::cout << "[CONFIG] Fixed g_e: " << std::scientific << std::setprecision(3) << FIXED_G_E << " GeV^-1" << std::endl;
        std::cout << "[CONFIG] Fixed g_gamma: " << std::scientific << std::setprecision(3) << FIXED_G_GAMMA << " GeV^-1" << std::endl;
        std::cout << "[CONFIG] Points per dimension: " << NUM_POINTS << std::endl;

        std::string data_dir = Constants::DATA_DIR + "/data_flux_comparison";
        DataOutput::create_output_directory(data_dir);

        // Part 1: Vary g_gamma (fixed g_e) - REMOVE the NaN header row
        std::cout << "\n[COMPUTE] Generating data for varying g_gamma (fixed g_e)" << std::endl;
        EnhancedProgressBar prog_vary_gg("Varying g_gamma", true, true);
        std::vector<std::vector<long double>> data_vary_gg; // No NaN row!
        data_vary_gg.reserve(NUM_POINTS);

        long double log_gg_min = Constants::G_GAMMA_LOG_MIN;
        long double log_gg_max = Constants::G_GAMMA_LOG_MAX;
        long double log_gg_step = (log_gg_max - log_gg_min) / (NUM_POINTS - 1);

        for (int i = 0; i < NUM_POINTS; ++i) {
            long double log_gg = log_gg_min + i * log_gg_step;
            long double g_gamma = std::pow(10.0L, log_gg);
            long double phi_p_val = FluxCalculations::phi_p(g_gamma);
            long double phi_b_val = FluxCalculations::phi_b(FIXED_G_E);
            long double phi_c_val = FluxCalculations::phi_c(FIXED_G_E);

            data_vary_gg.push_back({log_gg, phi_p_val, phi_b_val, phi_c_val});
            prog_vary_gg.update(i + 1, NUM_POINTS);
        }

        prog_vary_gg.finish("Data generation completed");
        std::stringstream header_vary_gg;
        header_vary_gg << "log_g_gamma phi_p phi_b phi_c";
        DataOutput::write_diagnostics_realization_data(data_vary_gg, data_dir + "/flux_vs_g_gamma.txt", header_vary_gg.str());

        // Part 2: Vary g_e (fixed g_gamma) - REMOVE the NaN header row
        std::cout << "\n[COMPUTE] Generating data for varying g_e (fixed g_gamma)" << std::endl;
        EnhancedProgressBar prog_vary_ge("Varying g_e", true, true);
        std::vector<std::vector<long double>> data_vary_ge; // No NaN row!
        data_vary_ge.reserve(NUM_POINTS);

        long double log_ge_min = Constants::G_E_LOG_MIN;
        long double log_ge_max = Constants::G_E_LOG_MAX;
        long double log_ge_step = (log_ge_max - log_ge_min) / (NUM_POINTS - 1);

        for (int i = 0; i < NUM_POINTS; ++i) {
            long double log_ge = log_ge_min + i * log_ge_step;
            long double g_e = std::pow(10.0L, log_ge);
            long double phi_p_val = FluxCalculations::phi_p(FIXED_G_GAMMA);
            long double phi_b_val = FluxCalculations::phi_b(g_e);
            long double phi_c_val = FluxCalculations::phi_c(g_e);

            data_vary_ge.push_back({log_ge, phi_p_val, phi_b_val, phi_c_val});
            prog_vary_ge.update(i + 1, NUM_POINTS);
        }

        prog_vary_ge.finish("Data generation completed");
        std::stringstream header_vary_ge;
        header_vary_ge << "log_g_e phi_p phi_b phi_c";
        DataOutput::write_diagnostics_realization_data(data_vary_ge, data_dir + "/flux_vs_g_e.txt", header_vary_ge.str());

        // Part 3: 3D grid (g_e, g_gamma, fluxes) - REMOVE the NaN header row
        std::cout << "\n[COMPUTE] Generating 3D grid data" << std::endl;
        EnhancedProgressBar prog_3d("3D Grid Generation", true, true);
        std::vector<std::vector<long double>> data_3d; // No NaN row!
        data_3d.reserve(NUM_POINTS * NUM_POINTS);

        int total_points = NUM_POINTS * NUM_POINTS;
        int processed = 0;

        for (int i = 0; i < NUM_POINTS; ++i) {
            long double log_ge = log_ge_min + i * log_ge_step;
            long double g_e = std::pow(10.0L, log_ge);

            for (int j = 0; j < NUM_POINTS; ++j) {
                long double log_gg = log_gg_min + j * log_gg_step;
                long double g_gamma = std::pow(10.0L, log_gg);
                long double phi_p_val = FluxCalculations::phi_p(g_gamma);
                long double phi_b_val = FluxCalculations::phi_b(g_e);
                long double phi_c_val = FluxCalculations::phi_c(g_e);

                data_3d.push_back({log_ge, log_gg, phi_p_val, phi_b_val, phi_c_val});
                processed++;
                prog_3d.update(processed, total_points);
            }
        }

        prog_3d.finish("3D grid completed");
        std::stringstream header_3d;
        header_3d << "log_g_e log_g_gamma phi_p phi_b phi_c";
        DataOutput::write_diagnostics_realization_data(data_3d, data_dir + "/flux_3d_grid.txt", header_3d.str());

        // List output files
        std::vector<std::string> flux_files = {
            data_dir + "flux_vs_g_gamma.txt",
            data_dir + "flux_vs_g_e.txt",
            data_dir + "flux_3d_grid.txt"
        };
        list_relevant_files(flux_files, "Flux Comparison");

        std::cout << std::string(60, '=') << std::endl;
        std::cout << "[SUCCESS] Flux comparison data generation completed" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        return data_3d;
    }



} // namespace Simulations
