#include "biot_savart.h"

Array biot_savart_dB_by_dX(Array& points, Array& gamma, Array& dgamma_by_dphi) {
    int num_points = points.shape(0);
    Array res = xt::zeros<double>({num_points, 3, 3});
    int num_quad_points = gamma.shape(0);
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points.at(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma.at(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi.at(j, 0));
            auto diff = point - gamma_j;
            auto norm_diff = norm(diff);
            auto norm_diff_4_inv = 1/(norm_diff*norm_diff*norm_diff*norm_diff);
            auto three_dgamma_by_dphi_cross_diff_by_norm_diff = cross(dgamma_by_dphi_j, diff) * 3 / norm_diff;

            ////Unrolling does not seem to help.
            //auto numerator1_0 = Vec3d{0., dgamma_by_dphi_j[2]*norm_diff, -dgamma_by_dphi_j[1]*norm_diff};//cross(dgamma_by_dphi_j, e1)*norm_diff;
            //auto numerator1_1 = Vec3d{-dgamma_by_dphi_j[2] * norm_diff, 0., dgamma_by_dphi_j[0] * norm_diff};//cross(dgamma_by_dphi_j, e2)*norm_diff;
            //auto numerator1_2 = Vec3d{dgamma_by_dphi_j[1] * norm_diff, -dgamma_by_dphi_j[0] * norm_diff, 0.};//cross(dgamma_by_dphi_j, e3)*norm_diff;
            //auto numerator2_0 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[0];
            //auto numerator2_1 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[1];
            //auto numerator2_2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[2];
            //auto temp_0 = (numerator1_0-numerator2_0)*norm_diff_4_inv;
            //auto temp_1 = (numerator1_1-numerator2_1)*norm_diff_4_inv;
            //auto temp_2 = (numerator1_2-numerator2_2)*norm_diff_4_inv;
            //res.at(i, 0, 0) += temp_0[0];
            //res.at(i, 1, 0) += temp_0[1];
            //res.at(i, 2, 0) += temp_0[2];
            //res.at(i, 0, 1) += temp_1[0];
            //res.at(i, 1, 1) += temp_1[1];
            //res.at(i, 2, 1) += temp_1[2];
            //res.at(i, 0, 2) += temp_2[0];
            //res.at(i, 1, 2) += temp_2[1];
            //res.at(i, 2, 2) += temp_2[2];

            for(int k=0; k<3; k++) {
                auto ek = Vec3d{0., 0., 0.};
                ek[k] = 1.0;
                auto numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
                auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                auto temp = (numerator1-numerator2) * norm_diff_4_inv;
                res.at(i, 0, k) += temp[0];
                res.at(i, 1, k) += temp[1];
                res.at(i, 2, k) += temp[2];
            }
        }
    }
    return res;
}
