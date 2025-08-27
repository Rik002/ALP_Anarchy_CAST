#ifndef DATA_OUTPUT_H
#define DATA_OUTPUT_H

#include "constants.h"
#include "matrix_operations.h"
#include <vector>
#include <string>
#include <memory>

namespace DataOutput {    // Enhanced data structures for comprehensive output

    struct StatisticalTestResults {
        double ks_statistic = 0.0;
        double ks_p_value = 0.0;
        bool ks_test_passed = false;
        double ad_statistic = 0.0;
        double ad_p_value = 0.0;
        bool ad_test_passed = false;
        double chi2_statistic = 0.0;
        double chi2_p_value = 0.0;
        bool chi2_test_passed = false;

        double sample_mean = 0.0;
        double sample_std = 0.0;
        double expected_mean = 0.0;
        double expected_std = 0.0;
        double relative_error_mean = 0.0;
        double relative_error_std = 0.0;

        int sample_size = 0;
        double validation_tolerance_used = 0.0;
    };

    struct MatrixDistributionData {
        int N = 0;
        int n_realizations = 0;

        std::vector<double> matrix_elements_gamma;
        std::vector<double> matrix_elements_e;
        std::vector<double> first_row_elements;
        std::vector<double> diagonal_elements;
        std::vector<double> off_diagonal_elements;
        std::vector<double> determinants;
        std::vector<double> determinant_errors;
        std::vector<double> orthogonality_errors;
        std::vector<double> matrix_norms;

        StatisticalTestResults element_statistics;
        StatisticalTestResults determinant_statistics;
        bool haar_measure_validated = false;
        double haar_quality_score = 0.0;
    };

    struct OscillationProbabilityData {
        int N = 0;
        int n_realizations = 0;

        std::vector<long double> p_gamma_gamma_values;
        std::vector<long double> p_e_gamma_values;
        std::vector<long double> p_gamma_e_values;
        std::vector<long double> p_e_e_values;

        StatisticalTestResults p_gamma_gamma_stats;
        StatisticalTestResults p_e_gamma_stats;

        double theoretical_p_gamma_gamma = 0.0;
        double theoretical_p_e_gamma = 0.0;
        bool n2_validation_passed = false;
        bool large_n_validation_passed = false;
    };

    struct CouplingDistributionData {
        int N = 0;
        int n_realizations = 0;

        std::vector<std::vector<double>> g_gamma_couplings;  // [realization][i]
        std::vector<std::vector<double>> g_e_couplings;      // [realization][i]
        std::vector<double> g_gamma_norms;
        std::vector<double> g_e_norms;

        StatisticalTestResults gamma_coupling_stats;
        StatisticalTestResults e_coupling_stats;

        bool coupling_conservation_passed = false;
        double max_coupling_conservation_error = 0.0;
    };

    struct ConvergenceTestData {
        std::vector<int> N_values;
        std::vector<int> realization_counts;
        std::vector<std::vector<double>> weights;           // [N_idx][realization_count_idx]
        std::vector<std::vector<long double>> p_gamma_means;     // [N_idx][realization_count_idx]
        std::vector<std::vector<long double>> p_gamma_stds;      // [N_idx][realization_count_idx]

        std::vector<bool> convergence_achieved;
        std::vector<int> min_realizations_needed;

        double convergence_criteria_relative_error = 0.01;  // 1% relative error
        double convergence_criteria_stability = 0.001;      // 0.1% change threshold
    };

    struct Figure2Data {
        int N = 0;
        int n_realizations = 0;

        std::vector<double> g_e_values;
        std::vector<double> g_gamma_values;
        std::vector<std::vector<double>> weight_matrix;          // [i][j]
        std::vector<std::vector<double>> p_gamma_mean_matrix;    // [i][j]
        std::vector<std::vector<double>> p_gamma_std_matrix;     // [i][j]
        std::vector<std::vector<double>> p_e_gamma_mean_matrix;  // [i][j]
        std::vector<std::vector<double>> p_e_gamma_std_matrix;   // [i][j]

        // Raw data for detailed analysis
        std::vector<std::vector<std::vector<double>>> p_gamma_raw_data;    // [i][j][realization]
        std::vector<std::vector<std::vector<double>>> p_e_gamma_raw_data;  // [i][j][realization]

        // Statistical test results
        std::vector<std::vector<StatisticalTestResults>> p_gamma_stats_matrix;    // [i][j]
        std::vector<std::vector<StatisticalTestResults>> p_e_gamma_stats_matrix;   // [i][j]
        std::vector<std::vector<std::vector<double>>> p_gamma_residuals;           // [i][j][realization]
        std::vector<std::vector<std::vector<double>>> p_e_gamma_residuals;        // [i][j][realization]

        // Global statistics
        StatisticalTestResults global_p_gamma_stats;
        StatisticalTestResults global_p_e_gamma_stats;

        double computation_time_seconds = 0.0;
        size_t total_realizations_computed = 0;

        // New field for W residuals
        std::vector<std::vector<std::vector<double>>> w_residuals; // Added for W - 0.5 residuals
        std::vector<std::vector<StatisticalTestResults>> w_stats_matrix; // Added for W Chi-Squared test
    };

    struct Figure3Data {
        int N = 0;
        int n_realizations = 0;
        long double g_e_fixed = 0.0;
        long double log_g_50gamma = 0.0;
        long double W = 0.0;
        StatisticalTestResults global_p_gamma_stats;
        StatisticalTestResults global_p_e_gamma_stats;
        std::vector<long double> p_gamma_raw_data;
        std::vector<long double> p_e_gamma_raw_data;
    };

    // Enhanced output functions
    bool write_matrix_distributions(const MatrixDistributionData& data, const std::string& base_filename);
    bool write_oscillation_distributions(const OscillationProbabilityData& data, const std::string& base_filename);
    bool write_coupling_distributions(const CouplingDistributionData& data, const std::string& base_filename);
    bool write_convergence_analysis(const ConvergenceTestData& data, const std::string& base_filename);
    bool write_figure2_comprehensive(const Figure2Data& data, const std::string& base_filename);
    bool write_figure3_comprehensive(const Figure3Data& data, const std::string& base_filename);
    bool write_statistical_summary(const std::vector<std::pair<std::string, StatisticalTestResults>>& results);

    // Updated function declaration
    bool write_figure2_stats(const Figure2Data& data, const std::string& base_filename);
    bool write_figure3_stats(const Figure3Data& data, const std::string& base_filename);

    // Original compatibility functions
    void write_diagnostics_realization_data(const std::vector<std::vector<long double>>& data,
                                           const std::string& filename,
                                           const std::string& header);
    void write_diagnostics_flux_data(const std::vector<std::vector<long double>>& data,
                                    const std::string& filename,
                                    const std::string& header);
    void write_matrix_distribution_data(const std::vector<std::vector<long double>>& data,
                                       const std::string& filename,
                                       const std::string& header);
    void write_figure_2_data(const std::vector<std::vector<long double>>& data,
                            const std::string& filename,
                            const std::string& header);
    void write_figure_3_data(const std::vector<std::vector<long double>>& data,
                            const std::string& filename,
                            const std::string& header);
    void write_convergence_fig2_data(const std::vector<std::vector<long double>>& data,
                                    const std::string& filename,
                                    const std::string& header);

    // Utility functions
    std::string format_duration(double seconds);
    std::string format_scientific(double value, int precision = 3);
    std::string format_percentage(double fraction, int decimal_places = 1);
    std::string generate_timestamp();
    std::string create_comprehensive_header(const std::string& data_type,
                                          const std::string& description,
                                          int N = 0, int n_realizations = 0);
    bool create_output_directory(const std::string& directory);

    // New utility function for printing tables
    void print_statistical_table(const std::vector<std::pair<std::string, StatisticalTestResults>>& results,
                               const std::string& title);
}

#endif
