#ifndef FLUX_CALCULATIONS_H
#define FLUX_CALCULATIONS_H

#include "constants.h"
#include <cmath>
#include <string>
#include <functional>

namespace FluxCalculations {

// Enhanced flux calculation functions with bounds checking
long double phi_p(long double g_gamma);
long double phi_b(long double g_e);
long double phi_c(long double g_e);

// CAST bound calculations with smooth transitions
long double g_gamma_n1(long double g_e);
long double phi_max_from_bound(long double g_gamma);

// Enhanced flux calculations with energy integration
long double integrated_phi_p(long double g_gamma, long double omega_min = 1.0, long double omega_max = 10.0);
long double integrated_phi_b(long double g_e, long double omega_min = 1.0, long double omega_max = 10.0);
long double integrated_phi_c(long double g_e, long double omega_min = 1.0, long double omega_max = 10.0);



long double integrate_flux_gsl(long double g_coupling, const std::string& flux_type);

// Corrected oscillated flux calculation (Equation 4.22)
long double oscillated_flux(long double g_gamma, long double g_e, long double p_gamma_gamma, long double p_e_gamma);

// Enhanced validation and diagnostic functions
bool validate_flux_calculations();
bool validate_coupling_parameters(long double g_gamma, long double g_e);

// Advanced flux analysis functions
struct FluxAnalysisResult {
    long double phi_p_value;
    long double phi_b_value;
    long double phi_c_value;
    long double phi_total;
    long double g_gamma_bound;
    bool parameters_valid;
    bool flux_finite;
    double relative_contributions[3]; // [Primakoff, Bremsstrahlung, Compton]
};

FluxAnalysisResult analyze_flux_components(long double g_gamma, long double g_e);

// Energy integration with adaptive quadrature
struct IntegrationResult {
    long double value;
    long double error_estimate;
    int function_evaluations;
    bool converged;
};

IntegrationResult integrate_flux_adaptive(
    long double g_coupling,
    const std::string& flux_type,
    long double omega_min = 1.0,
    long double omega_max = 10.0,
    long double tolerance = 1e-8
);

}

#endif
