#include "data_output.h"
#include <fstream>
#include <iomanip>

namespace DataOutput {
    void write_figure_2_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void write_figure_3_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void write_convergence_fig2_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void write_convergence_fig3_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void write_diagnostics_flux_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void write_diagnostics_realization_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void write_matrix_distribution_data(const std::vector<std::vector<long double>>& data, const std::string& filename, const std::string& header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        file << std::setprecision(18);
        file << "# " << header << "\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
    }
}
