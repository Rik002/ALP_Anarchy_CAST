#include "../include_cpp/flux_calculations.h"
#include "../include_cpp/constants.h"
#include <cmath>

namespace FluxCalculations {
    long double phi_p(long double g_gamma) {
        return Constants::PHI_P_COEFF * g_gamma * g_gamma;
    }

    long double phi_b(long double g_e) {
        return Constants::PHI_B_COEFF * g_e * g_e;
    }

    long double phi_c(long double g_e) {
        return Constants::PHI_C_COEFF * g_e * g_e;
    }

    long double g_gamma_n1(long double g_e) {
        return std::min(std::sqrt(Constants::G_GAMMA_G_E_LIMIT / g_e), Constants::G_GAMMA_N1);
    }
}
