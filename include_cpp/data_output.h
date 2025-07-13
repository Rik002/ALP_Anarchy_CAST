#ifndef DATA_OUTPUT_H
#define DATA_OUTPUT_H

#include <vector>
#include <string>

namespace DataOutput {
    void write_figure_2_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
    void write_figure_3_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
    void write_convergence_fig2_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
    void write_convergence_fig3_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
    void write_diagnostics_flux_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
    void write_diagnostics_realization_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
    void write_matrix_distribution_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header);
}

#endif
