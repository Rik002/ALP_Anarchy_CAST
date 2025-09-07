#include "../include_cpp/flux_calculations.h"
#include "../include_cpp/constants.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <gsl/gsl_integration.h>

namespace FluxCalculations {

long double phi_p(long double g_gamma) {
    if (g_gamma <= 0.0L || !std::isfinite(g_gamma)) return 0.0L;
    long double result = Constants::PHI_P_COEFF * g_gamma * g_gamma;
    return std::isfinite(result) ? result : 0.0L;
}

long double phi_b(long double g_e) {
    if (g_e <= 0.0L || !std::isfinite(g_e)) return 0.0L;
    long double result = Constants::PHI_B_COEFF * g_e * g_e;
    return std::isfinite(result) ? result : 0.0L;
}

long double phi_c(long double g_e) {
    if (g_e <= 0.0L || !std::isfinite(g_e)) return 0.0L;
    long double result = Constants::PHI_C_COEFF * g_e * g_e;
    return std::isfinite(result) ? result : 0.0L;
}

//long double g_gamma_n1(long double g_e) {
   // if (g_e <= 0.0L || !std::isfinite(g_e)) {
   //     return Constants::G_GAMMA_N1;
   // }
   // if (g_e < 1e-12L) {
   //     return Constants::G_GAMMA_N1;
   //  } else {
  //      return std::min(Constants::G_GAMMA_N1, Constants::G_GAMMA_G_E_LIMIT / g_e);
        //  }
//}

/*long double g_gamma_n1(long double g_e) {

    //Calculates the physically correct smooth transition bound for the single-ALP CAST constraint,
    //based on the analytical solution from the sum of fluxes.

    //This replaces the simple std::min() approach, which creates an unphysical sharp corner.
    //The formula is derived by solving (k_P * g_γ² + k_BCA * g_e²) * g_γ² = Signal_Limit for g_γ.


    // Ensure g_e is a positive, finite number to avoid math errors.
    if (g_e <= 0.0L || !std::isfinite(g_e)) {
        return Constants::G_GAMMA_N1;
    }

    // Define the constants from the established limits
    const long double G_GAMMA_P = Constants::G_GAMMA_N1;
    const long double G_GAMMA_GE_BCA = Constants::G_GAMMA_G_E_LIMIT;

    // The equation for the bound is a quadratic in g_γ²:
    // (g_γ²)² + b * (g_γ²) + c = 0
    // We solve it using the quadratic formula.

    // Constant 'a' for the quadratic formula, related to the signal limit from Primakoff
    const long double a_term = std::pow(G_GAMMA_P, 4);

    // Constant factor for the 'b' term in the quadratic formula
    const long double b_factor = std::pow(G_GAMMA_P, 4) / std::pow(G_GAMMA_GE_BCA, 2);

    // The 'b' term, which depends on g_e
    const long double b_term = b_factor * g_e * g_e;

    // Solve for g_gamma_squared, taking the positive root of the quadratic formula's discriminant
    const long double discriminant = std::sqrt(std::pow(b_term, 2) + 4.0L * a_term);
    const long double g_gamma_squared = (-b_term + discriminant) / 2.0L;

    // The final bound is the square root. Add a safety check for negative results from precision errors.
    if (g_gamma_squared < 0.0L) {
        return 0.0L;
    }

    return std::sqrt(g_gamma_squared);
}
*/

long double g_gamma_n1(long double g_e) {
    if (g_e <= 0.0L || !std::isfinite(g_e)) {
        return Constants::G_GAMMA_N1;
    }

    // Use more numerically stable approach
    const long double G_P = Constants::G_GAMMA_N1;
    const long double G_GE = Constants::G_GAMMA_G_E_LIMIT;

    // Avoid repeated pow() operations - pre-compute
    const long double G_P_sq = G_P * G_P;
    const long double G_P_quad = G_P_sq * G_P_sq;
    const long double G_GE_sq = G_GE * G_GE;
    const long double g_e_sq = g_e * g_e;

    // Use numerically stable quadratic formula
    const long double a = 1.0L;
    const long double b = (G_P_quad / G_GE_sq) * g_e_sq;
    const long double c = -G_P_quad;

    // For numerical stability, use the appropriate form based on b's magnitude
    long double g_gamma_sq;
    if (b > 0) {
        const long double sqrt_discriminant = std::sqrt(b * b - 4.0L * a * c);
        g_gamma_sq = (-b + sqrt_discriminant) / (2.0L * a);
    } else {
        g_gamma_sq = -c / (b + std::sqrt(b * b - 4.0L * a * c));
    }
    if(std::abs(std::log10(g_gamma_sq-G_P))<0.001) return G_P;
    else return (g_gamma_sq > 0.0L) ? std::sqrt(g_gamma_sq) : 0.0L;
}



long double integrated_phi_p(long double g_gamma, long double omega_min, long double omega_max) {
    const int n_points = 1000;
    long double delta_omega = (omega_max - omega_min) / n_points;
    long double total_flux = 0.0L;

    for (int i = 0; i <= n_points; ++i) {
        long double omega = omega_min + i * delta_omega;
        long double flux_elem = 2.0e-6L * g_gamma * g_gamma *
                                std::pow(omega, 2.450L) * std::exp(-0.829L * omega);

        if (i == 0 || i == n_points) {
            total_flux += 0.5L * flux_elem * delta_omega;
        } else {
            total_flux += flux_elem * delta_omega;
        }
    }
    return total_flux;
}

long double integrated_phi_b(long double g_e, long double omega_min, long double omega_max) {
    const int n_points = 1000;
    long double delta_omega = (omega_max - omega_min) / n_points;
    long double total_flux = 0.0L;

    for (int i = 0; i <= n_points; ++i) {
        long double omega = omega_min + i * delta_omega;
        long double flux_elem = 8.3e-6L * g_e * g_e * omega /
                                (1.0L + 0.667L * std::pow(omega, 1.278L)) * std::exp(-0.77L * omega);

        if (i == 0 || i == n_points) {
            total_flux += 0.5L * flux_elem * delta_omega;
        } else {
            total_flux += flux_elem * delta_omega;
        }
    }
    return total_flux;
}

long double integrated_phi_c(long double g_e, long double omega_min, long double omega_max) {
    const int n_points = 1000;
    long double delta_omega = (omega_max - omega_min) / n_points;
    long double total_flux = 0.0L;

    for (int i = 0; i <= n_points; ++i) {
        long double omega = omega_min + i * delta_omega;
        long double flux_elem = 4.2e-8L * g_e * g_e *
                                std::pow(omega, 2.987L) * std::exp(-0.776L * omega);

        if (i == 0 || i == n_points) {
            total_flux += 0.5L * flux_elem * delta_omega;
        } else {
            total_flux += flux_elem * delta_omega;
        }
    }
    return total_flux;
}

// This struct will pass the coupling parameter to the GSL integration function.
struct flux_params {
    long double g_coupling;
};

// GSL requires a C-style function. This wrapper calculates the differential Primakoff flux.
double phi_p_gsl_wrapper(double omega, void * p) {
    flux_params *params = static_cast<flux_params*>(p);
    long double g_gamma = params->g_coupling;
    if (omega <= 0) return 0.0;
    // Using the differential formula from the paper's image
    return 2.0e18 * std::pow(g_gamma / 1e-12L, 2) * std::pow(omega, 2.450L) * std::exp(-0.829L * omega);
}

// GSL wrapper for the differential Bremsstrahlung flux.
double phi_b_gsl_wrapper(double omega, void * p) {
    flux_params *params = static_cast<flux_params*>(p);
    long double g_e = params->g_coupling;
    if (omega <= 0) return 0.0;
    return 8.3e20 * std::pow(g_e / 1e-13L, 2) * (omega / (1.0L + 0.667L * std::pow(omega, 1.278L))) * std::exp(-0.77L * omega);
}

// GSL wrapper for the differential Compton flux.
double phi_c_gsl_wrapper(double omega, void * p) {
    flux_params *params = static_cast<flux_params*>(p);
    long double g_e = params->g_coupling;
    if (omega <= 0) return 0.0;
    return 4.2e18 * std::pow(g_e / 1e-13L, 2) * std::pow(omega, 2.987L) * std::exp(-0.776L * omega);
}

// The main high-accuracy integration function.
long double integrate_flux_gsl(long double g_coupling, const std::string& flux_type) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(10000);

    double result, error;
    gsl_function F;
    flux_params params = {g_coupling};
    F.params = &params;

    // Select the correct GSL wrapper based on the flux type
    if (flux_type == "phi_p") {
        F.function = &phi_p_gsl_wrapper;
    } else if (flux_type == "phi_b") {
        F.function = &phi_b_gsl_wrapper;
    } else if (flux_type == "phi_c") {
        F.function = &phi_c_gsl_wrapper;
    } else {
        gsl_integration_workspace_free(w);
        return 0.0L;
    }

    // Perform the integration over the CAST energy range [0.8, 6.8] keV
    // GSL_INTEG_GAUSS61 is a high-order rule. We ask for high relative accuracy (1e-7).
    gsl_integration_qag(&F, 1, 10, 0, 1e-8, 10000, GSL_INTEG_GAUSS61, w, &result, &error);

    gsl_integration_workspace_free(w);

    return static_cast<long double>(result);
}

long double phi_max_from_bound(long double g_gamma) {
    long double g_n1_at_g_gamma = g_gamma;
    return integrated_phi_p(g_n1_at_g_gamma,1.0L,10.0L);
}


long double oscillated_flux(long double g_gamma, long double g_e, long double p_gamma_gamma, long double p_e_gamma) {
    long double phi_gamma = phi_p(g_gamma);
    long double phi_e = phi_b(g_e) + phi_c(g_e);
    return p_gamma_gamma * phi_gamma + p_e_gamma * phi_e;
}

bool validate_flux_calculations() {
    const long double g_gamma_test = 1e-10;
    const long double g_e_test = 1e-13;

    long double test_phi_p = phi_p(g_gamma_test);
    long double test_phi_b = phi_b(g_e_test);
    long double test_phi_c = phi_c(g_e_test);

    if (test_phi_p <= 0 || test_phi_b <= 0 || test_phi_c <= 0) {
        std::cerr << "Error: Flux calculations return non-positive values" << std::endl;
        return false;
    }

    long double bound_test = g_gamma_n1(g_e_test);
    if (bound_test <= 0 || !std::isfinite(bound_test)) {
        std::cerr << "Error: g_gamma_n1 function invalid" << std::endl;
        return false;
    }
    return true;
}

bool validate_coupling_parameters(long double g_gamma, long double g_e) {
    // Check for non-positive or non-finite couplings
    if (g_gamma <= 0.0L || g_e <= 0.0L || !std::isfinite(g_gamma) || !std::isfinite(g_e)) {
        std::cerr << "Error: Invalid coupling parameters - g_gamma: " << g_gamma << ", g_e: " << g_e << std::endl;
        return false;
    }

    // Check if couplings are within physical bounds from constants
    if (g_gamma > Constants::SOLAR_NU_LIMIT) {
        std::cerr << "Error: g_gamma exceeds solar neutrino limit: " << g_gamma << " > " << Constants::SOLAR_NU_LIMIT << std::endl;
        return false;
    }

    // Check combined CAST constraint
    long double g_gamma_bound = g_gamma_n1(g_e);
    if (g_gamma > g_gamma_bound) {
        std::cerr << "Error: g_gamma exceeds CAST bound for g_e: " << g_gamma << " > " << g_gamma_bound << std::endl;
        return false;
    }

    // Check if couplings produce finite flux values
    long double test_phi_p = phi_p(g_gamma);
    long double test_phi_b = phi_b(g_e);
    long double test_phi_c = phi_c(g_e);

    if (!std::isfinite(test_phi_p) || !std::isfinite(test_phi_b) || !std::isfinite(test_phi_c)) {
        std::cerr << "Error: Non-finite flux values for g_gamma: " << g_gamma << ", g_e: " << g_e << std::endl;
        return false;
    }

    return true;
}

FluxAnalysisResult analyze_flux_components(long double g_gamma, long double g_e) {
    FluxAnalysisResult result = {0.0L, 0.0L, 0.0L, 0.0L, 0.0L, false, false, {0.0, 0.0, 0.0}};

    // Validate input parameters
    result.parameters_valid = validate_coupling_parameters(g_gamma, g_e);
    if (!result.parameters_valid) {
        return result;
    }

    // Calculate flux components
    result.phi_p_value = phi_p(g_gamma);
    result.phi_b_value = phi_b(g_e);
    result.phi_c_value = phi_c(g_e);
    result.phi_total = result.phi_p_value + result.phi_b_value + result.phi_c_value;

    // Check if flux values are finite
    result.flux_finite = std::isfinite(result.phi_p_value) &&
                         std::isfinite(result.phi_b_value) &&
                         std::isfinite(result.phi_c_value) &&
                         std::isfinite(result.phi_total);

    if (!result.flux_finite) {
        std::cerr << "Error: Non-finite flux components detected" << std::endl;
        return result;
    }

    // Calculate CAST bound
    result.g_gamma_bound = g_gamma_n1(g_e);

    // Calculate relative contributions
    if (result.phi_total > 0.0L) {
        result.relative_contributions[0] = static_cast<double>(result.phi_p_value / result.phi_total);
        result.relative_contributions[1] = static_cast<double>(result.phi_b_value / result.phi_total);
        result.relative_contributions[2] = static_cast<double>(result.phi_c_value / result.phi_total);
    } else {
        result.relative_contributions[0] = 0.0;
        result.relative_contributions[1] = 0.0;
        result.relative_contributions[2] = 0.0;
    }

    return result;
}

IntegrationResult integrate_flux_adaptive(
    long double g_coupling,
    const std::string& flux_type,
    long double omega_min,
    long double omega_max,
    long double tolerance) {

    IntegrationResult result = {0.0L, 0.0L, 0, false};

    // Select the appropriate flux function
    std::function<long double(long double)> flux_func;
    if (flux_type == "phi_p") {
        flux_func = [g_coupling](long double omega) {
            return Constants::PHI_P_COEFF * g_coupling * g_coupling *
                   std::pow(omega, 2.450L) * std::exp(-0.829L * omega);
        };
    } else if (flux_type == "phi_b") {
        flux_func = [g_coupling](long double omega) {
            return Constants::PHI_B_COEFF * g_coupling * g_coupling * omega /
                   (1.0L + 0.667L * std::pow(omega, 1.278L)) * std::exp(-0.77L * omega);
        };
    } else if (flux_type == "phi_c") {
        flux_func = [g_coupling](long double omega) {
            return Constants::PHI_C_COEFF * g_coupling * g_coupling *
                   std::pow(omega, 2.987L) * std::exp(-0.776L * omega);
        };
    } else {
        std::cerr << "Error: Unknown flux type: " << flux_type << std::endl;
        return result;
    }

    // Validate inputs
    if (g_coupling <= 0.0L || !std::isfinite(g_coupling) ||
        omega_min >= omega_max || !std::isfinite(omega_min) || !std::isfinite(omega_max) ||
        tolerance <= 0.0L) {
        std::cerr << "Error: Invalid integration parameters" << std::endl;
        return result;
    }

    // Adaptive Simpson's rule implementation
    const int max_iterations = 1000;
    const int min_intervals = 4;
    int n_intervals = min_intervals;
    long double previous_integral = 0.0L;
    int evaluations = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        long double h = (omega_max - omega_min) / n_intervals;
        long double integral = 0.0L;

        // Simpson's rule
        for (int i = 0; i <= n_intervals; ++i) {
            long double omega = omega_min + i * h;
            long double value = flux_func(omega);
            evaluations++;

            if (!std::isfinite(value)) {
                std::cerr << "Error: Non-finite flux value at omega=" << omega << std::endl;
                return result;
            }

            if (i == 0 || i == n_intervals) {
                integral += value / 3.0L;
            } else if (i % 2 == 0) {
                integral += 2.0L * value / 3.0L;
            } else {
                integral += 4.0L * value / 3.0L;
            }
        }
        integral *= h;

        // Estimate error
        result.error_estimate = std::abs(integral - previous_integral);
        result.value = integral;

        if (iter > 0 && result.error_estimate < tolerance) {
            result.converged = true;
            break;
        }

        previous_integral = integral;
        n_intervals *= 2; // Refine the grid
    }

    result.function_evaluations = evaluations;

    if (!result.converged) {
        std::cerr << "Warning: Adaptive integration did not converge within " << max_iterations << " iterations" << std::endl;
    }

    return result;
}

}
