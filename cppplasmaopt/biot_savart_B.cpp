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


template<class T>
void biot_savart_B_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& res) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto res_i   = Vec3dSimd();
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j          = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));

            auto diff = point_i - gamma_j;
            auto normdiff = normsq(diff);
            normdiff *= sqrt(normdiff);
            auto normdiffcubedinv = 1./normdiff;
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            res_i += dgamma_by_dphi_j_cross_diff * normdiffcubedinv;
        }
        for(int j=0; j<simd_size; j++){
            res(i+j, 0) = res_i.x[j];
            res(i+j, 1) = res_i.y[j];
            res(i+j, 2) = res_i.z[j];
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
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

template void biot_savart_B_simd<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

vector<Array> biot_savart_B_allcoils(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis) {
    int num_coils = gammas.size();
    auto res = vector<Array>();
    res.reserve(num_coils);
    long unsigned int num_points = points.shape(0);
    for(int i=0; i<num_coils; i++) {
        long unsigned int num_quad_points = gammas[i].shape(0);
        res.push_back(xt::pyarray<double>::from_shape({num_points, 3}));
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
       biot_savart_B_simd(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], res[i]);
       //biot_savart_B(points, gammas[i], dgamma_by_dphis[i], res[i]);
    }
    return res;
}
