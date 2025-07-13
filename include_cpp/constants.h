#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <unistd.h>

namespace Constants {
    inline long double G_GAMMA_N1;
    inline long double G_GAMMA_G_E_LIMIT;
    inline long double PHI_P_COEFF;
    inline long double PHI_B_COEFF;
    inline long double PHI_C_COEFF;
    inline long double SOLAR_NU_LIMIT;
    inline int N_REALIZATIONS;
    inline int MAX_N;
    inline int G_E_POINTS_PER_DECADE;
    inline int G_GAMMA_POINTS_PER_DECADE;
    inline int G_GAMMA_POINTS;
    inline long double G_E_LOG_MIN;
    inline long double G_E_LOG_MAX;
    inline long double G_GAMMA_LOG_MIN;
    inline long double G_GAMMA_LOG_MAX;
    inline std::string DATA_DIR;

    inline void load_constants(const std::string& default_filename = "../config/constants.dat") {
        std::string filename = default_filename;
        // Check for environment variable override
        if (const char* env_path = std::getenv("CONSTANTS_PATH")) {
            filename = env_path;
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            // Try fallback path
            filename = "./config/constants.dat";
            file.open(filename);
        }
        if (!file.is_open()) {
            char cwd[1024];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                throw std::runtime_error("Could not open constants.dat at " + filename +
                                         ". Current working directory: " + cwd +
                                         ". Check file existence or set CONSTANTS_PATH environment variable.");
            } else {
                throw std::runtime_error("Could not open constants.dat at " + filename +
                                         ". Check file existence or set CONSTANTS_PATH environment variable.");
            }
        }
        std::string line;
        bool in_constants_section = false;
        while (std::getline(file, line)) {
            // Skip empty lines, comments, or section headers
            if (line.empty() || line[0] == '#' || line[0] == ';' || line.find("-------") != std::string::npos) continue;
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
                throw std::runtime_error("Error parsing key '" + key + "' in constants.dat: " + e.what());
            }
        }
        file.close();
        // Compute G_GAMMA_POINTS
        G_GAMMA_POINTS = static_cast<int>(std::ceil((G_GAMMA_LOG_MAX - G_GAMMA_LOG_MIN) * G_GAMMA_POINTS_PER_DECADE));
        if (G_GAMMA_POINTS <= 0) {
            throw std::runtime_error("G_GAMMA_POINTS must be positive");
        }
        // Validate ranges and finiteness
        if (G_E_LOG_MIN >= G_E_LOG_MAX) {
            throw std::runtime_error("G_E_LOG_MIN must be less than G_E_LOG_MAX");
        }
        if (G_GAMMA_LOG_MIN >= G_GAMMA_LOG_MAX) {
            throw std::runtime_error("G_GAMMA_LOG_MIN must be less than G_GAMMA_LOG_MAX");
        }
        if (!std::isfinite(G_E_LOG_MIN) || !std::isfinite(G_E_LOG_MAX) ||
            !std::isfinite(G_GAMMA_LOG_MIN) || !std::isfinite(G_GAMMA_LOG_MAX)) {
            throw std::runtime_error("Constant values must be finite");
        }
    }
}

#endif
