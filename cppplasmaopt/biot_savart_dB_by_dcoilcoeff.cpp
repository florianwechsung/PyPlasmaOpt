#include "biot_savart.h"

Array biot_savart_dB_by_dcoilcoeff(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff) {
    int num_points = points.shape(0);
    int num_coil_coeffs = dgamma_by_dcoeff.shape(1);
    Array res = xt::zeros<double>({num_points, num_coil_coeffs, 3});
    int num_quad_points = gamma.shape(0);
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

            for(int k=0; k<num_coil_coeffs; k++) {
                auto d2gamma_j_by_dphi_dcoeff_k = Vec3d(3, &d2gamma_by_dphidcoeff.at(j, k, 0));
                auto dgamma_j_by_dcoeff_k = Vec3d(3, &dgamma_by_dcoeff.at(j, k, 0));

                auto term1 = norm_diff_3_inv * cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                auto term2 = norm_diff_3_inv * cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);
                auto term3 = norm_diff_5_inv * inner(dgamma_j_by_dcoeff_k, diff) * three_dgamma_by_dphi_cross_diff;
                auto temp = term1 - term2 + term3;
                res.at(i, k, 0) += temp[0];
                res.at(i, k, 1) += temp[1];
                res.at(i, k, 2) += temp[2];
            }
        }
    }
    return res;
}
