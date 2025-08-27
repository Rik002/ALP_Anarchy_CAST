#ifndef SIMULATIONS_H
#define SIMULATIONS_H

#include "constants.h"
#include "matrix_operations.h"
#include "data_output.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace Simulations {

// Enhanced result structures
struct RealizationResult {
    bool success; // bool
    long double p_gg; // long double
    long double p_eg; // long double
    long double p_gg_prime; // long double
    long double p_eg_prime; // long double
    long double phi_osc; // long double
    long double p_gg_p_value; // long double
    long double p_eg_p_value; // long double
    Eigen::VectorXd g_i_gamma; // Mass-basis gamma couplings
    Eigen::VectorXd g_i_e; // Mass-basis electron couplings
    long double det_err; // long double
    long double orth_err; // long double
    long double condition_number_gamma = 1.0L;
    long double condition_number_e = 1.0L;
    // Physics validation flags
    bool physics_valid = false;
    bool numerics_valid = false;
    // Diagnostic information
    std::string failure_reason;
};

struct PointResult {
    double weight = 0.0;
    double p_gamma_mean = 0.0;
    double p_gamma_std = 0.0;
    double p_e_gamma_mean = 0.0;
    double p_e_gamma_std = 0.0;
    std::vector<long double> p_gamma_values;
    std::vector<long double> p_e_gamma_values;
    std::vector<long double> p_gamma_e_values; // Added for completeness
    std::vector<long double> p_e_e_values; // Added for completeness
    MatrixOperations::MatrixStatistics matrix_stats;
    bool physics_validation_passed = false;
    bool statistics_converged = false;
    double validation_tolerance_used = 0.0;
    // Enhanced diagnostics
    int successful_realizations = 0;
    int failed_realizations = 0;
    std::vector<std::string> failure_reasons;
};

struct SimulationResults {
    std::vector<PointResult> figure2_results;
    std::vector<PointResult> figure3_results;
    MatrixOperations::MatrixStatistics overall_matrix_stats;
    double total_computation_time = 0.0;
    size_t peak_memory_usage = 0;
    size_t total_realizations_computed = 0;
    // Physics validation summary
    struct PhysicsValidationSummary {
        long double p_gamma_mean_N2 = 0.0;
        long double p_e_gamma_mean_N2 = 0.0;
        bool n2_validation_passed = false;
        std::vector<double> large_n_relative_errors;
        bool large_n_scaling_validated = false;
        double ks_p_value_elements = 0.0;
        bool haar_measure_validated = false;
        double fit_quality_slope = 0.0;
        double fit_quality_r_squared = 0.0;
        bool scaling_law_validated = false;
    } physics_summary;
    // Performance metrics
    struct PerformanceMetrics {
        double realizations_per_second = 0.0;
        double grid_points_per_second = 0.0;
        double average_realization_time_ms = 0.0;
        size_t memory_per_realization_bytes = 0;
        int threads_used = 1;
    } performance;
};

enum class SimulationMode {
    VALIDATION_ONLY,
    FIGURE_2_ONLY,
    FIGURE_3_ONLY,
    CONVERGENCE_ANALYSIS,
    FULL_ANALYSIS,
    DIAGNOSTIC_ONLY
};

class SimulationController {
private:
    bool validation_enabled_ = true;
    bool parallel_enabled_ = true;
    bool memory_optimized_ = true;
    bool detailed_logging_ = true;
    size_t peak_memory_usage_ = 0;
    SimulationResults results_;
    std::function<void(const std::string&, int, int)> progress_callback_;
    std::function<void(const std::string&, const std::string&)> log_callback_;

    void initialize_results();
    void finalize_results();
    void setup_parallel_processing();
    void update_progress(const std::string& stage, int current, int total);
    void log_message(const std::string& level, const std::string& message);
    // Memory management
    void monitor_memory_usage();
    void optimize_memory_usage();

public:
    SimulationController();
    ~SimulationController();
    // Configuration functions
    void set_validation_mode(bool enabled) { validation_enabled_ = enabled; }
    void set_parallel_mode(bool enabled) { parallel_enabled_ = enabled; }
    void set_memory_optimization(bool enabled) { memory_optimized_ = enabled; }
    void set_detailed_logging(bool enabled) { detailed_logging_ = enabled; }
    void set_progress_callback(std::function<void(const std::string&, int, int)> callback) {
        progress_callback_ = callback;
    }
    void set_log_callback(std::function<void(const std::string&, const std::string&)> callback) {
        log_callback_ = callback;
    }
    // Main simulation functions
    bool run_simulation(SimulationMode mode);
    bool run_validation_tests();
    bool generate_figure2_data(int N);
    bool generate_figure3_data();
    bool run_convergence_analysis();
    bool run_diagnostic_suite();
    // Results access
    const SimulationResults& get_results() const { return results_; }
    const MatrixOperations::MatrixStatistics& get_matrix_statistics() const { return results_.overall_matrix_stats; }
    size_t get_peak_memory_usage() const { return peak_memory_usage_; }
    // Enhanced analysis functions
    void perform_simultaneous_statistics(int N, int n_realizations,
                                        std::vector<long double>& p_gamma_values,
                                        std::vector<long double>& p_e_gamma_values,
                                        MatrixOperations::MatrixStatistics& matrix_stats);
    bool validate_physics_expectations(int N, const std::vector<long double>& p_gamma_values,
                                     const std::vector<long double>& p_e_gamma_values);
    DataOutput::Figure3Data analyze_scaling_law(const std::vector<int>& N_values,
                                               const std::vector<long double>& g_gamma_50_values);
};

// Enhanced core simulation functions
RealizationResult single_realization(int N, long double g_e, long double g_gamma, long double phi_max);
PointResult calculate_single_point(int N, double g_e, double g_gamma, int n_realizations,
                                  bool collect_statistics = false);
std::vector<double> generate_log_grid(double min_val, double max_val, int points_per_decade);

// MAIN SIMULATION FUNCTIONS - These are the core functions called from main()
std::vector<std::vector<long double>> generate_figure_2(int N);
std::vector<std::vector<long double>> generate_figure_3();

// MISSING FUNCTION DECLARATION - This was causing the compilation error!
/**
 * @brief Generate convergence analysis for multiple N values
 * @return Combined convergence data for all N values across different realization counts
 */
std::vector<std::vector<long double>> generate_convergence_analysis();

// DIAGNOSTIC FUNCTIONS
std::vector<std::vector<long double>> diagnose_flux();
std::vector<std::vector<long double>> diagnose_realization_data();
std::vector<std::vector<long double>> diagnose_matrix_distribution(int N);
std::vector<std::vector<long double>> generate_flux_comparison_data();

// DEPRECATED FUNCTIONS - These may exist but are replaced by generate_convergence_analysis()
std::vector<std::vector<long double>> generate_convergence_fig2();
std::vector<std::vector<long double>> generate_convergence_fig3();

// Enhanced diagnostic functions
bool validate_oscillation_probability_formulas();
bool validate_coupling_generation_equations();
bool validate_haar_measure_implementation();
bool validate_numerical_stability();

struct DiagnosticSummary {
    bool oscillation_formulas_correct = false;
    bool coupling_generation_correct = false;
    bool haar_measure_correct = false;
    bool numerical_stability_ok = false;
    bool physics_validation_passed = false;
    std::vector<std::string> errors_found;
    std::vector<std::string> warnings_issued;
    std::vector<std::string> recommendations;
    double overall_quality_score = 0.0; // 0-1 score
};

DiagnosticSummary run_comprehensive_diagnostics();

// Additional data structures for enhanced functionality
struct Figure2Data {
    int N = 0;
    int n_realizations = 0;
    std::vector<long double> g_e_values;
    std::vector<long double> g_gamma_values;
    std::vector<std::vector<double>> weight_matrix; // [i][j]
    std::vector<std::vector<double>> p_gamma_mean_matrix; // [i][j]
    std::vector<std::vector<double>> p_gamma_std_matrix; // [i][j]
    std::vector<std::vector<double>> p_e_gamma_mean_matrix; // [i][j]
    std::vector<std::vector<double>> p_e_gamma_std_matrix; // [i][j]
    // Raw data for detailed analysis
    std::vector<std::vector<std::vector<long double>>> p_gamma_raw_data; // [i][j][realization]
    std::vector<std::vector<std::vector<long double>>> p_e_gamma_raw_data; // [i][j][realization]
    // Statistical test results
    std::vector<std::vector<DataOutput::StatisticalTestResults>> p_gamma_stats_matrix; // [i][j]
    std::vector<std::vector<DataOutput::StatisticalTestResults>> p_e_gamma_stats_matrix; // [i][j]
    std::vector<std::vector<std::vector<long double>>> p_gamma_residuals; // [i][j][realization]
    std::vector<std::vector<std::vector<long double>>> p_e_gamma_residuals; // [i][j][realization]
    // Global statistics
    DataOutput::StatisticalTestResults global_p_gamma_stats;
    DataOutput::StatisticalTestResults global_p_e_gamma_stats;
    double computation_time_seconds = 0.0;
    size_t total_realizations_computed = 0;
    std::vector<std::vector<std::vector<long double>>> w_residuals; // Added for W - 0.5 residuals
    std::vector<std::vector<DataOutput::StatisticalTestResults>> w_stats_matrix; // Added for W Chi-Squared tests
};

} // namespace Simulations

#endif // SIMULATIONS_H
