#include "biot_savart.h"
#include <tuple>

std::pair<Array, Array> biot_savart_d2B_by_dXdcoilcoeff_via_chainrule(Array& points, Array& gamma, Array& dgamma_by_dphi) {
    int num_points = points.shape(0);
    int num_quad_points = gamma.shape(0);
    Array res_coil_gamma     = xt::zeros<double>({3, 3, 3, num_points, num_quad_points});
    Array res_coil_gammadash = xt::zeros<double>({3, 3, 3, num_points, num_quad_points});
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points.at(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j          = Vec3d(3, &gamma.at(j, 0));
            auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi.at(j, 0));

            auto diff      = point - gamma_j;
            auto norm_diff = norm(diff);

            auto norm_diff_3_inv           = 1/(norm_diff*norm_diff*norm_diff);
            auto norm_diff_5_inv           = norm_diff_3_inv/(norm_diff*norm_diff);
            auto norm_diff_7_inv           = norm_diff_5_inv/(norm_diff*norm_diff);
            auto dgamma_by_dphi_cross_diff = cross(dgamma_j_by_dphi, diff);

            for(int k=0; k<3; k++) { // iterate over the three pertubation directions of the curve
                auto ek = Vec3d{0., 0., 0.};
                ek.at(k) = 1.;
                for (int l = 0; l < 3; ++l) { // iterate over the three directions that the gradient is taken in
                    auto el  = Vec3d{0., 0., 0.};
                    el.at(l) = 1.0;
                    auto term1 = norm_diff_3_inv * cross(ek, el);
                    auto term2 = 3 * inner(diff, ek) * norm_diff_5_inv * cross(dgamma_j_by_dphi, el);
                    auto term3 = -15 * inner(diff, ek) * diff.at(l) * norm_diff_7_inv * dgamma_by_dphi_cross_diff;
                    auto term4 = 3 * ek.at(l) * norm_diff_5_inv * dgamma_by_dphi_cross_diff;
                    auto term5 = -3 * diff.at(l) * norm_diff_5_inv * cross(ek, diff);
                    auto term6 = 3 * diff.at(l) * norm_diff_5_inv * cross(dgamma_j_by_dphi, ek);
                    res_coil_gamma.at(k, l, 0, i, j) = (term2+term3+term4+term6).at(0);
                    res_coil_gamma.at(k, l, 1, i, j) = (term2+term3+term4+term6).at(1);
                    res_coil_gamma.at(k, l, 2, i, j) = (term2+term3+term4+term6).at(2);
                    res_coil_gammadash.at(k, l, 0, i, j) = (term1+term5).at(0);
                    res_coil_gammadash.at(k, l, 1, i, j) = (term1+term5).at(1);
                    res_coil_gammadash.at(k, l, 2, i, j) = (term1+term5).at(2);
                }
            }
        }
    }
    return std::make_pair(res_coil_gamma, res_coil_gammadash);
}
