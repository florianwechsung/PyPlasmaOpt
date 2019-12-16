#include "biot_savart.h"

void biot_savart_dB_by_dX(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& res) {
    int num_points = points.shape(0);
    int num_quad_points = gamma.shape(0);
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point - gamma_j;
            auto norm_diff = norm(diff);
            auto norm_diff_4_inv = 1/(norm_diff*norm_diff*norm_diff*norm_diff);
            auto three_dgamma_by_dphi_cross_diff_by_norm_diff = cross(dgamma_by_dphi_j, diff) * 3 / norm_diff;

            for(int k=0; k<3; k++) {
                auto ek = Vec3d{0., 0., 0.};
                ek[k] = 1.0;
                auto numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
                auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                auto temp = (numerator1-numerator2) * norm_diff_4_inv;
                res(i, k, 0) += temp[0];
                res(i, k, 1) += temp[1];
                res(i, k, 2) += temp[2];
            }
        }
    }
}
