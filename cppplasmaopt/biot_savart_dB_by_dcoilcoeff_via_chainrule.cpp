#include "biot_savart.h"
#include <tuple>


std::pair<Array, Array> biot_savart_dB_by_dcoilcoeff_via_chainrule(Array& points, Array& gamma, Array& dgamma_by_dphi) {
    int num_points      = points.shape(0);
    int num_quad_points = gamma.shape(0);
    Array res_coil_gamma     = xt::zeros<double>({num_points, num_quad_points, 3, 3});
    Array res_coil_gammadash = xt::zeros<double>({num_points, num_quad_points, 3, 3});

    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points.at(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma.at(j, 0));
            auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi.at(j, 0));
            auto diff = point - gamma_j;
            auto norm_diff = norm(diff);
            auto norm_diff_3_inv = 1/(norm_diff*norm_diff*norm_diff);
            auto norm_diff_5_inv = norm_diff_3_inv/(norm_diff*norm_diff);
            auto three_dgamma_by_dphi_cross_diff = cross(dgamma_j_by_dphi, diff) * 3;
            for(int k=0; k<3; k++) {
                auto ek = Vec3d{0., 0., 0.};
                ek[k] = 1.0;
                auto term1 = norm_diff_3_inv * cross(ek, diff);
                auto term2 = norm_diff_3_inv * cross(dgamma_j_by_dphi, ek);
                auto term3 = norm_diff_5_inv * inner(ek, diff) * three_dgamma_by_dphi_cross_diff;
                res_coil_gamma.at(i, j, k, 0) = (-term2 + term3).at(0);
                res_coil_gamma.at(i, j, k, 1) = (-term2 + term3).at(1);
                res_coil_gamma.at(i, j, k, 2) = (-term2 + term3).at(2);
                res_coil_gammadash.at(i, j, k, 0) = term1.at(0);
                res_coil_gammadash.at(i, j, k, 1) = term1.at(1);
                res_coil_gammadash.at(i, j, k, 2) = term1.at(2);
            }
        }
    }
    return std::make_pair(res_coil_gamma, res_coil_gammadash);
}
