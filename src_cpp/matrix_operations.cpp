#include "../include_cpp/matrix_operations.h"
#include "../include_cpp/constants.h"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>  // Required for some GSL operations
#include <gsl/gsl_errno.h> // For GSL_SUCCESS and error handling
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_histogram.h>


namespace MatrixOperations {

// Thread-local random number generators
thread_local std::mt19937 RandomNumberManager::generator(std::random_device{}());
std::random_device RandomNumberManager::rd;

void RandomNumberManager::seed(unsigned int s) {
    generator.seed(s);
}

double RandomNumberManager::uniform_real(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    double result;
    int max_attempts = 10;
    for (int i = 0; i < max_attempts; ++i) {
        result = dist(generator);
        if (std::isfinite(result)) return result;
        std::cerr << "[WARNING] Non-finite random number generated, retrying (" << i+1 << "/" << max_attempts << ")" << std::endl;
    }
    std::cerr << "[ERROR] Failed to generate finite random number after " << max_attempts << " attempts" << std::endl;
    return a; // Fallback to lower bound
}

double RandomNumberManager::normal(double mean, double std) {
    std::normal_distribution<double> dist(mean, std);
    return dist(generator);
}

int RandomNumberManager::uniform_int(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(generator);
}

double RandomNumberManager::haar_angle(int dimension) {
    if (dimension == 2) {
        return uniform_real(0.0, 2.0 * M_PI);
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double u = uniform(generator);
    return std::acos(std::pow(u, 1.0 / (dimension - 1)));
}

Eigen::MatrixXd random_so_n(int N, MatrixMethod method) {
    switch (method) {
        case MatrixMethod::EXACT_HAAR:
            return generate_so_n_inductive(N);
        case MatrixMethod::GIVENS_ROTATIONS:
            return generate_so_n_givens(N);
        default:
            return generate_so_n_qr(N);
    }
}

Eigen::MatrixXd generate_so_n_qr(int N) {
    Eigen::MatrixXd A(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i, j) = RandomNumberManager::normal(0.0, 1.0);
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    // Apply sign correction for proper SO(N) distribution
    Eigen::VectorXd signs = R.diagonal().cwiseSign();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(signs);
    Q = Q * D;

    // Ensure determinant is +1
    if (Q.determinant() < 0) {
        Q.col(0) *= -1;
    }
    return Q;
}

Eigen::MatrixXd generate_so_n_inductive(int N) {
    // Exact Haar measure using inductive construction
    if (N == 1) {
        Eigen::MatrixXd result(1, 1);
        result(0, 0) = 1.0;
        return result;
    }
    if (N == 2) {
        double theta = RandomNumberManager::uniform_real(0.0, 2.0 * M_PI);
        Eigen::MatrixXd result(2, 2);
        result(0, 0) = std::cos(theta);
        result(0, 1) = -std::sin(theta);
        result(1, 0) = std::sin(theta);
        result(1, 1) = std::cos(theta);
        if (!result.allFinite()) {
            std::cerr << "[ERROR] Non-finite elements in SO(2) matrix" << std::endl;
            return Eigen::MatrixXd::Identity(2, 2); // Fallback to identity
        }
        return result;
    }

    // For N > 2, use inductive construction
    Eigen::MatrixXd U_prev = generate_so_n_inductive(N - 1);

    // Embed U_{N-1} in N dimensions
    Eigen::MatrixXd S_N = Eigen::MatrixXd::Identity(N, N);
    S_N.topLeftCorner(N - 1, N - 1) = U_prev;

    // Generate angles with proper Haar measure
    std::vector<double> angles(N - 1);
    for (int i = 0; i < N - 2; ++i) {
        angles[i] = RandomNumberManager::haar_angle(N - i);
    }
    angles[N - 2] = RandomNumberManager::uniform_real(0.0, 2.0 * M_PI);

    // Construct rotation matrix
    Eigen::MatrixXd U_prime = Eigen::MatrixXd::Identity(N, N);
    for (int k = N - 2; k >= 0; --k) {
        Eigen::MatrixXd R_k = Eigen::MatrixXd::Identity(N, N);
        R_k(k, k) = std::cos(angles[k]);
        R_k(k, N - 1) = -std::sin(angles[k]);
        R_k(N - 1, k) = std::sin(angles[k]);
        R_k(N - 1, N - 1) = std::cos(angles[k]);
        U_prime = U_prime * R_k;
    }
    Eigen::MatrixXd result = U_prime * S_N;
    if (!result.allFinite()) {
        std::cerr << "[ERROR] Non-finite elements in SO(" << N << ") matrix" << std::endl;
        return Eigen::MatrixXd::Identity(N, N);
    }
    return result;
}

Eigen::MatrixXd generate_so_n_givens(int N) {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(N, N);
    int num_givens = N * (N - 1) / 2;
    for (int k = 0; k < num_givens; ++k) {
        // Random angles for each Givens rotation
        double theta = RandomNumberManager::haar_angle(N);
        // Select random pair (i, j)
        int i = RandomNumberManager::uniform_int(0, N - 2);
        int j = RandomNumberManager::uniform_int(i + 1, N - 1);
        // Construct Givens rotation G
        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(N, N);
        G(i, i) = std::cos(theta);
        G(j, j) = std::cos(theta);
        G(i, j) = -std::sin(theta);
        G(j, i) = std::sin(theta);
        // Accumulate rotation
        Q = G * Q;
    }
    // Ensure determinant is +1 for SO(N)
    if (Q.determinant() < 0) {
        Q.col(0) *= -1;
    }
    return Q;
}

// Oscillation probability P_gamma_gamma (Equation 4.18)
long double calculate_p_gamma_gamma(const Eigen::VectorXd& g_i_gamma) {
    if (g_i_gamma.size() == 0) return 0.0L;
    long double sum2 = 0.0L, sum4 = 0.0L;
    for (int i = 0; i < g_i_gamma.size(); ++i) {
        long double g_i = g_i_gamma(i);
        if (!std::isfinite(g_i)) {
            std::cerr << "[ERROR] Non-finite coupling g_i_gamma[" << i << "] = " << g_i << std::endl;
            return 0.0L;
        }
        long double g_i_sq = g_i * g_i;
        sum2 += g_i_sq;
        sum4 += g_i_sq * g_i_sq;
    }
    if (!std::isfinite(sum2)) {
        std::cerr << "[ERROR] Non-finite sum2 = " << sum2 << " in P_gamma_gamma" << std::endl;
        return 0.0L;
    }
    // Proceed with P_gamma_gamma calculation
    long double result = sum4 / (sum2 * sum2);
    if (!std::isfinite(result)) {
        std::cerr << "[ERROR] Non-finite P_gamma_gamma = " << result << std::endl;
        return 0.0L;
    }
    return result;
}

// Oscillation probability P_e_gamma (Equation 4.20)
long double calculate_p_e_gamma(const Eigen::VectorXd& g_i_e, const Eigen::VectorXd& g_i_gamma)
{
    // Compute: numerator = sum_i (g_i_e^2 * g_i_gamma^2)
    // denominator = (sum_i g_i_e^2) * (sum_i g_i_gamma^2)
    long double num = 0.0, denom_e = 0.0, denom_g = 0.0;
    int N = g_i_e.size();
    for(int i = 0; i < N; ++i) {
        num += g_i_e(i)*g_i_e(i) * g_i_gamma(i)*g_i_gamma(i);
        denom_e += g_i_e(i)*g_i_e(i);
        denom_g += g_i_gamma(i)*g_i_gamma(i);
    }
    long double result = (denom_e > 0 && denom_g > 0) ? (num / (denom_e * denom_g)) : 0.0;
    if (!std::isfinite(result) || result < 0 || result > 1) {
        return 0.0;
    }
    return result;
}


// Coupling generation (Equation 3.3)
std::pair<Eigen::VectorXd, Eigen::VectorXd> generate_couplings(
    int N, long double g_gamma_total, long double g_e_total,
    const Eigen::MatrixXd& U_gamma, const Eigen::MatrixXd& U_e) {
    try {
        if (!U_gamma.allFinite() || !U_e.allFinite()) {
            throw std::runtime_error("Non-finite input matrices");
        }
        Eigen::VectorXd base_gamma = Eigen::VectorXd::Zero(N);
        Eigen::VectorXd base_e = Eigen::VectorXd::Zero(N);
        base_gamma(0) = g_gamma_total;
        base_e(0) = g_e_total;
        Eigen::VectorXd g_i_gamma = U_gamma * base_gamma;
        Eigen::VectorXd g_i_e = U_e * base_e;
        if (!g_i_gamma.allFinite() || !g_i_e.allFinite()) {
            throw std::runtime_error("Non-finite coupling values generated");
        }
        return std::make_pair(g_i_gamma, g_i_e);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Matrix operation failed: " << e.what() << std::endl;
        std::cerr << "[DEBUG] U_gamma:\n" << U_gamma << "\nU_e:\n" << U_e << std::endl;
        return std::make_pair(Eigen::VectorXd::Zero(N), Eigen::VectorXd::Zero(N));
    }
}

// Matrix validation functions
long double matrix_determinant_error(const Eigen::MatrixXd& matrix) {
    if (!matrix.allFinite()) return std::numeric_limits<long double>::max();
    long double det = matrix.determinant();
    if (!std::isfinite(det)) return std::numeric_limits<long double>::max();
    return std::abs(det - 1.0);
}

long double matrix_orthogonality_error(const Eigen::MatrixXd& matrix) {
    if (!matrix.allFinite()) return std::numeric_limits<long double>::max();
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
    Eigen::MatrixXd product = matrix * matrix.transpose();
    if (!product.allFinite()) return std::numeric_limits<long double>::max();
    long double norm = (product - identity).norm();
    if (!std::isfinite(norm)) return std::numeric_limits<long double>::max();
    return norm;
}

bool validate_so_n_matrix(const Eigen::MatrixXd& matrix, long double tolerance) {
    return matrix_determinant_error(matrix) < tolerance &&
           matrix_orthogonality_error(matrix) < tolerance;
}

// Legacy functions for compatibility
long double p_gamma_gamma(const Eigen::VectorXd& g_i_gamma) {
    return calculate_p_gamma_gamma(g_i_gamma);
}

long double p_e_gamma(const Eigen::VectorXd& g_i_e, const Eigen::VectorXd& g_i_gamma) {
    return calculate_p_e_gamma(g_i_e, g_i_gamma);
}

std::pair<double, double> kolmogorov_smirnov_test(const std::vector<long double>& data, const std::string& distribution) {
    if (data.empty()) return {0.0, 1.0};
    std::vector<double> data_double(data.begin(), data.end());
    std::sort(data_double.begin(), data_double.end());
    size_t n = data_double.size();
    double mean = gsl_stats_mean(data_double.data(), 1, n);
    double sd = gsl_stats_sd(data_double.data(), 1, n);
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = data_double[i];
        double empirical_cdf = static_cast<double>(i + 1) / n;
        double theoretical_cdf = gsl_cdf_ugaussian_P((x - mean) / sd);
        double diff = std::abs(empirical_cdf - theoretical_cdf);
        max_diff = std::max(max_diff, diff);
    }
    double ks_stat = max_diff * std::sqrt(static_cast<double>(n));
    // Approximate p-value for KS test (using asymptotic distribution)
    double lambda = ks_stat * (1.0 + 0.12 / std::sqrt(static_cast<double>(n)) + 0.11 / n);
    double p_value = std::exp(-2.0 * lambda * lambda); // Simplified asymptotic approximation; for accuracy, use GSL if available
    return {ks_stat, p_value};
}



std::pair<double, double> chi_square_gof_test(const std::vector<long double>& data, const std::string& distribution) {
    if (data.empty() || data.size() < Constants::MIN_SAMPLES_FOR_KS) {
        return {0.0, 1.0}; // Insufficient data
    }
    std::vector<double> sorted_data(data.begin(), data.end());
    std::sort(sorted_data.begin(), sorted_data.end());
    double mean = gsl_stats_mean(sorted_data.data(), 1, sorted_data.size());
    double sd = gsl_stats_sd(sorted_data.data(), 1, sorted_data.size());
    if (sd <= 0) return {0.0, 1.0};
    size_t n_bins = std::max(static_cast<size_t>(5), static_cast<size_t>(std::sqrt(sorted_data.size())));
    double min_val = sorted_data.front();
    double max_val = sorted_data.back();
    gsl_histogram* hist = gsl_histogram_alloc(n_bins);
    gsl_histogram_set_ranges_uniform(hist, min_val, max_val + 1e-10);
    for (double x : sorted_data) {
        gsl_histogram_increment(hist, x);
    }
    std::vector<double> expected(n_bins, 0.0);
    double total = static_cast<double>(sorted_data.size());
    for (size_t i = 0; i < n_bins; ++i) {
        double bin_lower, bin_upper;
        gsl_histogram_get_range(hist, i, &bin_lower, &bin_upper);
        double z1 = (bin_lower - mean) / sd;
        double z2 = (bin_upper - mean) / sd;
        double p1 = gsl_cdf_ugaussian_P(z1);
        double p2 = gsl_cdf_ugaussian_P(z2);
        expected[i] = total * (p2 - p1);
    }
    double chi2 = 0.0;
    for (size_t i = 0; i < n_bins; ++i) {
        double observed = gsl_histogram_get(hist, i);
        if (expected[i] > 0) {
            double diff = observed - expected[i];
            chi2 += (diff * diff) / expected[i];
        }
    }
    int df = static_cast<int>(n_bins) - 3; // For normal dist (mean, sd fitted)
    if (df <= 0) {
        gsl_histogram_free(hist);
        return {0.0, 1.0};
    }
    double p_value = gsl_cdf_chisq_Q(chi2, static_cast<double>(df)); // Survival function (1 - CDF)
    gsl_histogram_free(hist);
    return {chi2, p_value};
}



double kolmogorov_smirnov_arcsine_test(const std::vector<double>& data) {
        if (data.size() < static_cast<size_t>(Constants::MIN_SAMPLES_FOR_KS)) {
            return 0.0; // Not enough samples
        }

        std::vector<double> sorted_data(data);
        std::sort(sorted_data.begin(), sorted_data.end());

        double d_max = 0.0;
        size_t n = sorted_data.size();

        for (size_t i = 0; i < n; ++i) {
            double x = sorted_data[i];
            // Arcsine CDF: F(x) = (2/π) * arcsin(√x) for x in [0,1]
            double cdf_value = (x >= 0.0 && x <= 1.0) ? (2.0 / M_PI) * std::asin(std::sqrt(x)) : (x < 0.0 ? 0.0 : 1.0);
            double empirical_cdf = static_cast<double>(i + 1) / n;
            double d1 = std::abs(empirical_cdf - cdf_value);
            double d2 = std::abs((static_cast<double>(i) / n) - cdf_value);
            d_max = std::max(d_max, std::max(d1, d2));
        }

        // Approximate p-value using GSL
        double p_value = 1.0 - gsl_cdf_ugaussian_P(d_max * std::sqrt(static_cast<double>(n)));
        return p_value >= Constants::KS_ALPHA_LEVEL ? d_max : d_max;
    }

}
