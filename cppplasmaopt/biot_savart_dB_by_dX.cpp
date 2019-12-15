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

template<class T>
void biot_savart_dB_by_dX_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& res) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto res_i_dx   = Vec3dSimd();
        auto res_i_dy   = Vec3dSimd();
        auto res_i_dz   = Vec3dSimd();
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j          = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point_i - gamma_j;
            auto norm_diff_2     = normsq(diff);
            auto norm_diff       = sqrt(norm_diff_2);
            auto norm_diff_4_inv = 1/(norm_diff_2*norm_diff_2);
            auto three_dgamma_by_dphi_cross_diff_by_norm_diff = cross(dgamma_by_dphi_j, diff) * (3/norm_diff);
            // k = 0
            auto ek = Vec3dSimd(1., 0., 0.);
            auto numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
            auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff.x;
            auto temp = (numerator1-numerator2) * norm_diff_4_inv;
            res_i_dx += temp;
            // k = 1
            ek = Vec3dSimd(0., 1., 0.);
            numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
            numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff.y;
            temp = (numerator1-numerator2) * norm_diff_4_inv;
            res_i_dy += temp;
            // k = 2
            ek = Vec3dSimd(0., 0., 1.);
            numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
            numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff.z;
            temp = (numerator1-numerator2) * norm_diff_4_inv;
            res_i_dz += temp;
        }
        for(int j=0; j<simd_size; j++){
            res(i+j, 0, 0) = res_i_dx.x[j];
            res(i+j, 0, 1) = res_i_dx.y[j];
            res(i+j, 0, 2) = res_i_dx.z[j];
            res(i+j, 1, 0) = res_i_dy.x[j];
            res(i+j, 1, 1) = res_i_dy.y[j];
            res(i+j, 1, 2) = res_i_dy.z[j];
            res(i+j, 2, 0) = res_i_dz.x[j];
            res(i+j, 2, 1) = res_i_dz.y[j];
            res(i+j, 2, 2) = res_i_dz.z[j];
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma.at(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi.at(j, 0));
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
template void biot_savart_dB_by_dX_simd<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

vector<Array> biot_savart_dB_by_dX_allcoils(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis) {
    int num_coils = gammas.size();
    auto res = vector<Array>();
    res.reserve(num_coils);
    int num_points = points.shape(0);
    for(int i=0; i<num_coils; i++) {
        res.push_back(xt::zeros<double>({num_points, 3, 3}));
    }
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
       biot_savart_dB_by_dX_simd(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], res[i]);
       //biot_savart_dB_by_dX(points, gammas[i], dgamma_by_dphis[i], res[i]);
    }
    return res;
}
