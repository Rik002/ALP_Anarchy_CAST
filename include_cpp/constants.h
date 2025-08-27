#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <filesystem>
#include <functional>  // FIX: Added missing header for std::function

namespace Constants {
    // Progress callback type definition
    using ProgressCallback = std::function<void(const std::string&, int, int, double)>;

    // Physical Constants
    inline long double G_GAMMA_N1;
    inline long double G_GAMMA_G_E_LIMIT;
    inline long double PHI_P_COEFF;
    inline long double PHI_B_COEFF;
    inline long double PHI_C_COEFF;
    inline long double SOLAR_NU_LIMIT;

    // Simulation Parameters
    inline int N_REALIZATIONS;
    inline int MAX_N;
    inline int N_THREADS;

    // Statistical Testing Parameters - ENHANCEMENT: Adaptive validation criteria
    inline double KS_ALPHA_LEVEL = 0.05;
    inline double MATRIX_TOLERANCE = 0;
    inline int MIN_SAMPLES_FOR_KS = 30;

    // ENHANCEMENT: Adaptive physics validation tolerances
    inline double PHYSICS_TOLERANCE_STRICT = 0.10;   // 10% for N >= 10000 realizations
    inline double PHYSICS_TOLERANCE_MEDIUM = 0.20;   // 20% for 1000 <= N < 10000 realizations
    inline double PHYSICS_TOLERANCE_LOOSE = 0.50;    // 50% for N < 1000 realizations

    // Grid Parameters
    inline int G_E_POINTS_PER_DECADE;
    inline int G_GAMMA_POINTS_PER_DECADE;
    inline int G_GAMMA_POINTS;
    inline long double G_E_LOG_MIN;
    inline long double G_E_LOG_MAX;
    inline long double G_GAMMA_LOG_MIN;
    inline long double G_GAMMA_LOG_MAX;

    // Output Configuration
    inline std::string DATA_DIR;

    // ENHANCEMENT: Adaptive validation tolerance function
    inline double get_validation_tolerance(int n_realizations) {
        if (n_realizations < 1000) {
            return PHYSICS_TOLERANCE_LOOSE;    // 50% tolerance for small samples
        } else if (n_realizations < 10000) {
            return PHYSICS_TOLERANCE_MEDIUM;   // 20% tolerance for medium samples
        } else {
            return PHYSICS_TOLERANCE_STRICT;   // 10% tolerance for large samples
        }
    }

    // Load constants from file
    inline bool load_from_file(const std::string& filename = "config/constants.dat") {
        std::ifstream file(filename);
        if (!file.is_open()) {
            // Try alternative paths
            std::string alt_filename = "../config/constants.dat";
            file.open(alt_filename);
            if (!file.is_open()) {
                alt_filename = "./config/constants.dat";
                file.open(alt_filename);
                if (!file.is_open()) {
                    std::cerr << "Error: Could not open constants.dat at any location" << std::endl;
                    return false;
                }
            }
        }

        std::string line;
        bool in_constants_section = false;

        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';' ||
                line.find("-------") != std::string::npos) continue;

            if (line.find("[Constants]") != std::string::npos) {
                in_constants_section = true;
                continue;
            }

            if (!in_constants_section) continue;

            // Parse key = value : description
            size_t eq_pos = line.find('=');
            size_t colon_pos = line.find(':');
            if (eq_pos == std::string::npos || colon_pos == std::string::npos) continue;

            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1, colon_pos - eq_pos - 1);

            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            try {
                if (key == "G_GAMMA_N1") G_GAMMA_N1 = std::stold(value);
                else if (key == "G_GAMMA_G_E_LIMIT") G_GAMMA_G_E_LIMIT = std::stold(value);
                else if (key == "PHI_P_COEFF") PHI_P_COEFF = std::stold(value);
                else if (key == "PHI_B_COEFF") PHI_B_COEFF = std::stold(value);
                else if (key == "PHI_C_COEFF") PHI_C_COEFF = std::stold(value);
                else if (key == "SOLAR_NU_LIMIT") SOLAR_NU_LIMIT = std::stold(value);
                else if (key == "N_REALIZATIONS") {
                    N_REALIZATIONS = std::stoi(value);
                    if (N_REALIZATIONS <= 0) throw std::runtime_error("N_REALIZATIONS must be positive");
                }
                else if (key == "MAX_N") {
                    MAX_N = std::stoi(value);
                    if (MAX_N < 2) throw std::runtime_error("MAX_N must be at least 2");
                }
                else if (key == "N_THREADS") {
                    N_THREADS = std::stoi(value);
                    if (N_THREADS <= 0) N_THREADS = 1;
                }
                else if (key == "G_E_POINTS_PER_DECADE") {
                    G_E_POINTS_PER_DECADE = std::stoi(value);
                    if (G_E_POINTS_PER_DECADE <= 0) throw std::runtime_error("G_E_POINTS_PER_DECADE must be positive");
                }
                else if (key == "G_GAMMA_POINTS_PER_DECADE") {
                    G_GAMMA_POINTS_PER_DECADE = std::stoi(value);
                    if (G_GAMMA_POINTS_PER_DECADE <= 0) throw std::runtime_error("G_GAMMA_POINTS_PER_DECADE must be positive");
                }
                else if (key == "G_E_LOG_MIN") G_E_LOG_MIN = std::stold(value);
                else if (key == "G_E_LOG_MAX") G_E_LOG_MAX = std::stold(value);
                else if (key == "G_GAMMA_LOG_MIN") G_GAMMA_LOG_MIN = std::stold(value);
                else if (key == "G_GAMMA_LOG_MAX") G_GAMMA_LOG_MAX = std::stold(value);
                else if (key == "DATA_DIR") DATA_DIR = value;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing key '" << key << "': " << e.what() << std::endl;
                return false;
            }
        }

        file.close();

        // Compute derived quantities
        G_GAMMA_POINTS = static_cast<int>(std::ceil((G_GAMMA_LOG_MAX - G_GAMMA_LOG_MIN) * static_cast<long double>(G_GAMMA_POINTS_PER_DECADE)));
        if (G_GAMMA_POINTS <= 0) {
            std::cerr << "Error: G_GAMMA_POINTS must be positive" << std::endl;
            return false;
        }

        // Validate ranges
        if (G_E_LOG_MIN >= G_E_LOG_MAX || G_GAMMA_LOG_MIN >= G_GAMMA_LOG_MAX) {
            std::cerr << "Error: Invalid parameter ranges" << std::endl;
            return false;
        }

        return true;
    }

    // Legacy function for compatibility
    inline void load_constants(const std::string& filename = "../config/constants.dat") {
        if (!load_from_file(filename)) {
            throw std::runtime_error("Failed to load constants from " + filename);
        }
    }
}

#endif
