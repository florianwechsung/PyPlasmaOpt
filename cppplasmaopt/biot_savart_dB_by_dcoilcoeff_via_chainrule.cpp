#include "biot_savart.h"
#include <tuple>

Array biot_savart_dB_by_dcoilcoeff_via_chainrule(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff) {
    int num_points      = points.shape(0);
    int num_quad_points = gamma.shape(0);
    int num_coeffs      = dgamma_by_dcoeff.shape(1);
    Array res_coil_gamma     = xt::zeros<double>({3, 3, num_points, num_quad_points});
    Array res_coil_gammadash = xt::zeros<double>({3, 3, num_points, num_quad_points});

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
                res_coil_gamma.at(k, 0, i, j) = (-term2 + term3).at(0);
                res_coil_gamma.at(k, 1, i, j) = (-term2 + term3).at(1);
                res_coil_gamma.at(k, 2, i, j) = (-term2 + term3).at(2);
                res_coil_gammadash.at(k, 0, i, j) = term1.at(0);
                res_coil_gammadash.at(k, 1, i, j) = term1.at(1);
                res_coil_gammadash.at(k, 2, i, j) = term1.at(2);
            }
        }
    }
    RowMat res_coil_0(num_points, num_coeffs);
    RowMat res_coil_1(num_points, num_coeffs);
    RowMat res_coil_2(num_points, num_coeffs);
    int j = 0;
    res_coil_0 = RowMat(num_points, num_quad_points, &res_coil_gamma.at(j, 0, 0, 0)) * ColMat(num_quad_points, num_coeffs, &dgamma_by_dcoeff.at(0, 0, j)) + RowMat(num_points, num_quad_points, &res_coil_gammadash.at(j, 0, 0, 0)) * ColMat(num_quad_points, num_coeffs, &d2gamma_by_dphidcoeff.at(0, 0, j));
    res_coil_1 = RowMat(num_points, num_quad_points, &res_coil_gamma.at(j, 1, 0, 0)) * ColMat(num_quad_points, num_coeffs, &dgamma_by_dcoeff.at(0, 0, j)) + RowMat(num_points, num_quad_points, &res_coil_gammadash.at(j, 1, 0, 0)) * ColMat(num_quad_points, num_coeffs, &d2gamma_by_dphidcoeff.at(0, 0, j));
    res_coil_2 = RowMat(num_points, num_quad_points, &res_coil_gamma.at(j, 2, 0, 0)) * ColMat(num_quad_points, num_coeffs, &dgamma_by_dcoeff.at(0, 0, j)) + RowMat(num_points, num_quad_points, &res_coil_gammadash.at(j, 2, 0, 0)) * ColMat(num_quad_points, num_coeffs, &d2gamma_by_dphidcoeff.at(0, 0, j));
    for (j = 1; j < 3; ++j) {
        res_coil_0 += RowMat(num_points, num_quad_points, &res_coil_gamma.at(j, 0, 0, 0)) * ColMat(num_quad_points, num_coeffs, &dgamma_by_dcoeff.at(0, 0, j)) + RowMat(num_points, num_quad_points, &res_coil_gammadash.at(j, 0, 0, 0)) * ColMat(num_quad_points, num_coeffs, &d2gamma_by_dphidcoeff.at(0, 0, j));
        res_coil_1 += RowMat(num_points, num_quad_points, &res_coil_gamma.at(j, 1, 0, 0)) * ColMat(num_quad_points, num_coeffs, &dgamma_by_dcoeff.at(0, 0, j)) + RowMat(num_points, num_quad_points, &res_coil_gammadash.at(j, 1, 0, 0)) * ColMat(num_quad_points, num_coeffs, &d2gamma_by_dphidcoeff.at(0, 0, j));
        res_coil_2 += RowMat(num_points, num_quad_points, &res_coil_gamma.at(j, 2, 0, 0)) * ColMat(num_quad_points, num_coeffs, &dgamma_by_dcoeff.at(0, 0, j)) + RowMat(num_points, num_quad_points, &res_coil_gammadash.at(j, 2, 0, 0)) * ColMat(num_quad_points, num_coeffs, &d2gamma_by_dphidcoeff.at(0, 0, j));
    }
    Array res = xt::zeros<double>({num_points, num_coeffs, 3});
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_coeffs; ++j) {
           res.at(i, j, 0) = res_coil_0(i, j); 
           res.at(i, j, 1) = res_coil_1(i, j); 
           res.at(i, j, 2) = res_coil_2(i, j); 
        }
    }
    return res;
}
