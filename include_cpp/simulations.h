#ifndef SIMULATIONS_H
#define SIMULATIONS_H

#include <vector>
#include <Eigen/Dense>

namespace Simulations {
    struct RealizationResult {
        bool success;
        long double p_gg;
        long double p_eg;
        long double phi_osc;
        Eigen::VectorXd u_i0_gamma;
        Eigen::VectorXd u_i0_e;
    };

    RealizationResult single_realization(int N, long double g_e, long double g_gamma, long double phi_max);
    std::vector<std::vector<long double>> generate_figure_2(int N);
    std::vector<std::vector<long double>> generate_figure_3();
    std::vector<std::vector<long double>> generate_convergence_fig2();
    std::vector<std::vector<long double>> generate_convergence_fig3();
    std::vector<std::vector<long double>> diagnose_flux();
    std::vector<std::vector<long double>> diagnose_realization_data();
    std::vector<std::vector<long double>> diagnose_matrix_distribution(int N);
}

#endif
