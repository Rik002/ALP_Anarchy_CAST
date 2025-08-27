#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "constants.h"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <functional>
#include <string>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_histogram.h>

namespace MatrixOperations {

// Enhanced matrix generation methods
enum class MatrixMethod {
QR_APPROXIMATION, // Fast but approximate
EXACT_HAAR, // Exact Haar measure (recommended)
GIVENS_ROTATIONS, // Alternative exact method
INDUCTIVE_CONSTRUCTION // Appendix A method
};

// Progress callback for batch operations
using MatrixProgressCallback = std::function<void(int current, int total, double progress)>;

// Enhanced random number management
class RandomNumberManager {

private:

  static thread_local std::mt19937 generator;
  static std::random_device rd;

public:

  static void seed(unsigned int s = 0);
  static double uniform_real(double a = 0.0, double b = 1.0);
  static double normal(double mean = 0.0, double std = 1.0);
  static int uniform_int(int a, int b);
  static double haar_angle(int dimension);
  static std::vector<double> random_unit_vector(int N);
};

// CORRECTED matrix generation functions

Eigen::MatrixXd random_so_n(int N, MatrixMethod method = MatrixMethod::EXACT_HAAR);
Eigen::MatrixXd generate_so_n_qr(int N);
Eigen::MatrixXd generate_so_n_inductive(int N); // Exact Haar from Appendix A
Eigen::MatrixXd generate_so_n_givens(int N);

// CORRECTED oscillation probability calculations (Equations 4.18 and 4.20)

long double calculate_p_gamma_gamma(const Eigen::VectorXd& g_i_gamma);
long double calculate_p_e_gamma(const Eigen::VectorXd& g_i_e, const Eigen::VectorXd& g_i_gamma);
long double calculate_p_gamma_e(const Eigen::VectorXd& g_i_gamma, const Eigen::VectorXd& g_i_e);
long double calculate_p_e_e(const Eigen::VectorXd& g_i_e);

// CORRECTED coupling generation (Equation 3.3)

std::pair<Eigen::VectorXd, Eigen::VectorXd> generate_couplings(

int N, long double g_gamma_total, long double g_e_total,

const Eigen::MatrixXd& U_gamma, const Eigen::MatrixXd& U_e);

// Enhanced matrix validation functions

long double matrix_determinant_error(const Eigen::MatrixXd& matrix);
long double matrix_orthogonality_error(const Eigen::MatrixXd& matrix);
long double matrix_condition_number(const Eigen::MatrixXd& matrix);
bool validate_so_n_matrix(const Eigen::MatrixXd& matrix, long double tolerance = Constants::MATRIX_TOLERANCE);

// Advanced statistical analysis structures

struct AdvancedStatistics {

double mean = 0.0;
double variance = 0.0;
double std_dev = 0.0;
double q25 = 0.0; // 25th percentile
double median = 0.0; // 50th percentile
double q75 = 0.0; // 75th percentile
double iqr = 0.0; // Interquartile range
double min_value = 0.0;
double max_value = 0.0;
double range = 0.0;
double skewness = 0.0;
double kurtosis = 0.0;
double excess_kurtosis = 0.0;

// Statistical tests

double ks_statistic = 0.0;
double ks_p_value = 0.0;
bool ks_test_passed = false;
double ad_statistic = 0.0;
double ad_p_value = 0.0;
bool ad_test_passed = false;

// Theoretical comparisons

double expected_mean = 0.0;
double expected_std = 0.0;
double mean_relative_error = 0.0;
double std_relative_error = 0.0;

// Outlier analysis

std::vector<int> outlier_indices;
int num_outliers = 0;
double outlier_threshold = 3.0; // Standard deviations

};

// Comprehensive matrix statistics structure

struct MatrixStatistics {

// Raw data collections

 std::vector<double> matrix_elements;
 std::vector<double> diagonal_elements;
 std::vector<double> off_diagonal_elements;
 std::vector<double> first_row_elements;
 std::vector<double> determinant_errors;
 std::vector<double> orthogonality_errors;
 std::vector<double> matrix_norms_frobenius;
 std::vector<double> matrix_norms_spectral;
 std::vector<double> condition_numbers;

// Computed statistics

AdvancedStatistics element_stats;
AdvancedStatistics diagonal_stats;
AdvancedStatistics off_diagonal_stats;
AdvancedStatistics first_row_stats;
AdvancedStatistics determinant_stats;
AdvancedStatistics orthogonality_stats;

// Matrix quality metrics

double max_determinant_error = 0.0;
double max_orthogonality_error = 0.0;
int num_invalid_matrices = 0;
double fraction_valid_matrices = 1.0;

// Haar measure validation

struct HaarValidation {
bool elements_uniform = false;
double element_ks_p_value = 0.0;
bool first_row_uniform = false;
double first_row_ks_p_value = 0.0;
bool determinants_valid = false;
bool orthogonality_valid = false;
double haar_quality_score = 0.0;

} haar_validation;

// Physics validation

struct PhysicsValidation {
bool n2_case_validated = false;
bool large_n_scaling = false;
double theoretical_element_std = 0.0;
double measured_element_std = 0.0;
double scaling_relative_error = 0.0;
bool coupling_conservation = false;
double coupling_conservation_error = 0.0;

} physics_validation;

void clear();
void compute_comprehensive_statistics(int N);
void validate_haar_measure(int N);
void validate_physics(int N, double expected_coupling_total_gamma = 0.0,
double expected_coupling_total_e = 0.0);

};

// Advanced statistical functions

AdvancedStatistics compute_advanced_statistics(const std::vector<double>& data,
const std::string& distribution_type = "normal",
double expected_mean = 0.0,
double expected_std = 1.0);

// Statistical test functions

std::pair<double, double> kolmogorov_smirnov_test(const std::vector<long double>& data, const std::string& distribution = "normal");
double kolmogorov_smirnov_arcsine_test(const std::vector<double>& data); // New function for arcsine distribution
double anderson_darling_test(const std::vector<double>& data, const std::string& distribution = "normal");
std::pair<double, double> chi_square_gof_test(const std::vector<long double>& data, const std::string& distribution = "normal");

// Physics validation functions

bool validate_n2_case(const std::vector<double>& p_gamma_values,
                      const std::vector<double>& p_e_gamma_values,
                      double tolerance = Constants::PHYSICS_TOLERANCE_MEDIUM);
bool validate_large_n_scaling(const std::vector<double>& p_gamma_values, int N,
                               double tolerance = Constants::PHYSICS_TOLERANCE_LOOSE);
bool validate_coupling_conservation(const Eigen::VectorXd& original_couplings,
                                    const Eigen::VectorXd& generated_couplings,
                                    double tolerance = Constants::PHYSICS_TOLERANCE_STRICT);

// Utility functions for statistical analysis

std::vector<double> compute_quantiles(std::vector<double> data,
const std::vector<double>& percentiles);
double compute_skewness(const std::vector<double>& data);
double compute_kurtosis(const std::vector<double>& data);
std::vector<int> detect_outliers(const std::vector<double>& data, double threshold = 3.0);

// Batch processing functions for performance

std::vector<Eigen::MatrixXd> generate_matrix_batch(int N, int count,
                                                    MatrixMethod method = MatrixMethod::EXACT_HAAR,
                                                    MatrixProgressCallback progress_cb = nullptr);
MatrixStatistics analyze_matrix_batch(const std::vector<Eigen::MatrixXd>& matrices,
                                      MatrixProgressCallback progress_cb = nullptr);

// Thread-safe matrix generator for parallel processing

class ThreadSafeMatrixGenerator {
private:

 MatrixMethod method_;
 int batch_size_;

public:

  ThreadSafeMatrixGenerator(MatrixMethod method = MatrixMethod::EXACT_HAAR, int batch_size = 100);
  std::vector<Eigen::MatrixXd> generate_batch(int N, int count);
  Eigen::MatrixXd generate_single(int N);

};

// Legacy functions for compatibility

long double p_gamma_gamma(const Eigen::VectorXd& g_i_gamma);
long double p_e_gamma(const Eigen::VectorXd& g_i_e, const Eigen::VectorXd& g_i_gamma);

// Added for oscillation condition

inline double DELTA_M2_MIN = 1e-12;

}

#endif

