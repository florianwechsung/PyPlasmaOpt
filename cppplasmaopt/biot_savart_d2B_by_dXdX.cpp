#include "biot_savart.h"

void biot_savart_d2B_by_dXdX(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& res) {
    int num_points = points.shape(0);
    int num_quad_points = gamma.shape(0);
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points.at(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma.at(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi.at(j, 0));
            auto diff = point - gamma_j;
            auto norm_diff = norm(diff);
            auto norm_diff_5_inv = 1/(norm_diff*norm_diff*norm_diff*norm_diff*norm_diff);
            auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff*norm_diff);
            auto dgamma_by_dphi_cross_diff = cross(dgamma_by_dphi_j, diff);

            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<3; k2++) {
                    auto ek1 = Vec3d{0., 0., 0.};
                    ek1[k1] = 1.0;
                    auto ek2 = Vec3d{0., 0., 0.};
                    ek2[k2] = 1.0;

                    auto term1 = -3 * (diff[k1]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, ek2);
                    auto term2 = -3 * (diff[k2]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, ek1);
                    auto term3 = 15 * (diff[k1] * diff[k2] * norm_diff_7_inv) * dgamma_by_dphi_cross_diff;
                    auto term4 = Vec3d{0., 0., 0.};
                    if(k1 == k2) {
                        term4 = -3 * norm_diff_5_inv * dgamma_by_dphi_cross_diff;
                    }
                    auto temp = (term1 + term2 + term3 + term4);
                    res.at(i, k1, k2, 0) += temp[0];
                    res.at(i, k1, k2, 1) += temp[1];
                    res.at(i, k1, k2, 2) += temp[2];
                }
            }
        }
    }
}
