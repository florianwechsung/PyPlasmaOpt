#include "biot_savart.h"

void biot_savart_B(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& res) {
    int num_points = points.shape(0);
    int num_quad_points = gamma.shape(0);
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points(i, 0));
        res(i, 0) = 0;
        res(i, 1) = 0;
        res(i, 2) = 0;
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point - gamma_j;
            double norm_diff = norm(diff);
            auto temp = cross(dgamma_by_dphi_j, diff) / (norm_diff * norm_diff * norm_diff);
            res(i, 0) += temp[0];
            res(i, 1) += temp[1];
            res(i, 2) += temp[2];
        }
    }
}
