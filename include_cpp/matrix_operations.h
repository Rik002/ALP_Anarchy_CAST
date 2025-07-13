#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <Eigen/Dense>

namespace MatrixOperations {
    Eigen::MatrixXd random_so_n(int N);
    long double p_gamma_gamma(const Eigen::VectorXd& g_i_gamma);
    long double p_e_gamma(const Eigen::VectorXd& g_i_e, const Eigen::VectorXd& g_i_gamma);
}

#endif
