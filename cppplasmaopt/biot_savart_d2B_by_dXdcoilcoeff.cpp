#include "biot_savart.h"

void biot_savart_d2B_by_dXdcoilcoeff(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff, Array& res) {
    int num_points = points.shape(0);
    int num_coil_coeffs = dgamma_by_dcoeff.shape(1);
    int num_quad_points = gamma.shape(0);
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j          = Vec3d(3, &gamma(j, 0));
            auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi(j, 0));

            auto diff      = point - gamma_j;
            auto norm_diff = norm(diff);

            auto norm_diff_3_inv           = 1/(norm_diff*norm_diff*norm_diff);
            auto norm_diff_5_inv           = norm_diff_3_inv/(norm_diff*norm_diff);
            auto norm_diff_7_inv           = norm_diff_5_inv/(norm_diff*norm_diff);
            auto dgamma_by_dphi_cross_diff = cross(dgamma_j_by_dphi, diff);

            for(int k=0; k<num_coil_coeffs; k++) {
                auto d2gamma_j_by_dphi_dcoeff_k                   = Vec3d(3, &d2gamma_by_dphidcoeff(j, k, 0));
                auto dgamma_j_by_dcoeff_k                         = Vec3d(3, &dgamma_by_dcoeff(j, k, 0));
                auto diff_inner_dgamma_j_by_dcoeff_k              = inner(diff, dgamma_j_by_dcoeff_k);
                Vec3d d2gamma_j_by_dphi_dcoeff_k_cross_diff       = cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                Vec3d dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k = cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);
                for (int l = 0; l < 3; ++l) {
                    auto el  = Vec3d{0., 0., 0.};
                    el[l] = 1.0;
                    auto term1 = norm_diff_3_inv * cross(d2gamma_j_by_dphi_dcoeff_k, el);
                    auto term2 = 3 * diff_inner_dgamma_j_by_dcoeff_k * norm_diff_5_inv * cross(dgamma_j_by_dphi, el);
                    auto term3 = -15 * diff_inner_dgamma_j_by_dcoeff_k * diff[l] * norm_diff_7_inv * dgamma_by_dphi_cross_diff;
                    auto term4 = 3 * dgamma_j_by_dcoeff_k[l] * norm_diff_5_inv * dgamma_by_dphi_cross_diff;
                    auto term5 = -3 * diff[l] * norm_diff_5_inv * d2gamma_j_by_dphi_dcoeff_k_cross_diff;
                    auto term6 = 3 * diff[l] * norm_diff_5_inv * dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k;
                    auto temp = term1 + term2 + term3 + term4 + term5 + term6;
                    res(i, k, 0, l) += temp[0];
                    res(i, k, 1, l) += temp[1];
                    res(i, k, 2, l) += temp[2];
                }
            }
        }
    }
}
