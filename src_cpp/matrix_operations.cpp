#include "../include_cpp/matrix_operations.h"
#include <random>
#include <Eigen/Dense>

namespace MatrixOperations {
    Eigen::MatrixXd random_so_n(int N) {
        thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<long double> dist(0.0L, 1.0L);
        Eigen::MatrixXd A(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A(i, j) = dist(gen);
            }
        }
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        Eigen::MatrixXd Q = qr.householderQ();
        return Q;
    }

    long double p_gamma_gamma(const Eigen::VectorXd& g_i_gamma) {
        long double sum = g_i_gamma.squaredNorm();
        return sum > 0.0L ? g_i_gamma(0) * g_i_gamma(0) / sum : 0.0L;
    }

    long double p_e_gamma(const Eigen::VectorXd& g_i_e, const Eigen::VectorXd& g_i_gamma) {
        long double sum = g_i_gamma.squaredNorm();
        return sum > 0.0L ? g_i_e(0) * g_i_gamma(0) / sum : 0.0L;
    }
}
