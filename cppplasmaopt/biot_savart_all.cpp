#include "biot_savart.h"

template<class T>
void biot_savart_all_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto B_i   = Vec3dSimd();
        //auto dB_dX_i = vector<Vec3dSimd>{
        auto dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };
        //auto d2B_dXdX_i = vector<Vec3dSimd>{
        auto d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd() 
        };
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j          = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));

            auto diff = point_i - gamma_j;
            auto norm_diff_2     = normsq(diff);
            auto norm_diff       = sqrt(norm_diff_2);

            auto norm_diff_3_inv = 1./(norm_diff_2 * norm_diff);
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            B_i += dgamma_by_dphi_j_cross_diff * norm_diff_3_inv;

            auto norm_diff_4_inv = 1/(norm_diff_2*norm_diff_2);
            auto three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff * (3/norm_diff);
            for(int k=0; k<3; k++) {
                auto ek = Vec3dSimd(0., 0., 0.);
                ek[k] += 1.;
                auto numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
                auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                auto temp = (numerator1-numerator2) * norm_diff_4_inv;
                dB_dX_i[k] += temp;
            }

            auto norm_diff_5_inv = norm_diff_4_inv/norm_diff;
            auto norm_diff_7_inv = norm_diff_5_inv/norm_diff_2;
            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<=k1; k2++) {
                    auto ek1 = Vec3dSimd(0., 0., 0.);
                    ek1[k1] += 1.;
                    auto ek2 = Vec3dSimd(0., 0., 0.);
                    ek2[k2] += 1.;

                    auto term1 =  cross(dgamma_by_dphi_j, ek2) * ((-3.) * (diff[k1]*norm_diff_5_inv));
                    auto term2 =  cross(dgamma_by_dphi_j, ek1) * ((-3.) * (diff[k2]*norm_diff_5_inv));
                    auto term3 =  dgamma_by_dphi_j_cross_diff * (15. * (diff[k1] * diff[k2] * norm_diff_7_inv));
                    auto term4 = Vec3dSimd(0., 0., 0.);
                    if(k1 == k2) {
                        term4 += dgamma_by_dphi_j_cross_diff * ((-3.) * norm_diff_5_inv);
                    }
                    d2B_dXdX_i[3*k1 + k2] += term1 + term2 + term3 + term4;
                }
            }
        }

        for(int j=0; j<simd_size; j++){
            B(i+j, 0) = B_i.x[j];
            B(i+j, 1) = B_i.y[j];
            B(i+j, 2) = B_i.z[j];
            for(int k=0; k<3; k++) {
                dB_by_dX(i+j, k, 0) = dB_dX_i[k].x[j];
                dB_by_dX(i+j, k, 1) = dB_dX_i[k].y[j];
                dB_by_dX(i+j, k, 2) = dB_dX_i[k].z[j];
            }
            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<=k1; k2++) {
                    d2B_by_dXdX(i+j, k1, k2, 0) = d2B_dXdX_i[3*k1 + k2].x[j];
                    d2B_by_dXdX(i+j, k1, k2, 1) = d2B_dXdX_i[3*k1 + k2].y[j];
                    d2B_by_dXdX(i+j, k1, k2, 2) = d2B_dXdX_i[3*k1 + k2].z[j];
                    if(k2 < k1){
                        d2B_by_dXdX(i+j, k2, k1, 0) = d2B_dXdX_i[3*k1 + k2].x[j];
                        d2B_by_dXdX(i+j, k2, k1, 1) = d2B_dXdX_i[3*k1 + k2].y[j];
                        d2B_by_dXdX(i+j, k2, k1, 2) = d2B_dXdX_i[3*k1 + k2].z[j];
                    }
                }
            }
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        B(i, 0) = 0;
        B(i, 1) = 0;
        B(i, 2) = 0;
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point - gamma_j;
            double norm_diff = norm(diff);
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            auto B_i = dgamma_by_dphi_j_cross_diff / (norm_diff * norm_diff * norm_diff);

            B(i, 0) += B_i[0];
            B(i, 1) += B_i[1];
            B(i, 2) += B_i[2];
            auto norm_diff_4_inv = 1/(norm_diff*norm_diff*norm_diff*norm_diff);
            auto three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff * 3 / norm_diff;
            for(int k=0; k<3; k++) {
                auto ek = Vec3d{0., 0., 0.};
                ek[k] = 1.0;
                auto numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
                auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                auto temp = (numerator1-numerator2) * norm_diff_4_inv;
                dB_by_dX(i, k, 0) += temp[0];
                dB_by_dX(i, k, 1) += temp[1];
                dB_by_dX(i, k, 2) += temp[2];
            }
            auto norm_diff_5_inv = norm_diff_4_inv/norm_diff;
            auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff*norm_diff);
            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<3; k2++) {
                    auto ek1 = Vec3d{0., 0., 0.};
                    ek1[k1] = 1.0;
                    auto ek2 = Vec3d{0., 0., 0.};
                    ek2[k2] = 1.0;

                    auto term1 = -3 * (diff[k1]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, ek2);
                    auto term2 = -3 * (diff[k2]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, ek1);
                    auto term3 = 15 * (diff[k1] * diff[k2] * norm_diff_7_inv) * dgamma_by_dphi_j_cross_diff;
                    auto term4 = Vec3d{0., 0., 0.};
                    if(k1 == k2) {
                        term4 = -3 * norm_diff_5_inv * dgamma_by_dphi_j_cross_diff;
                    }
                    auto temp = (term1 + term2 + term3 + term4);
                    d2B_by_dXdX(i, k1, k2, 0) += temp[0];
                    d2B_by_dXdX(i, k1, k2, 1) += temp[1];
                    d2B_by_dXdX(i, k1, k2, 2) += temp[2];
                }
            }
        }
    }
}
template void biot_savart_all_simd<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);


void biot_savart_all(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& B, Array& dB_by_dX, Array& d2B_by_dXdX, vector<Array>& dB_by_coilcurrents, vector<Array>& d2B_by_dXdcoilcurrents) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    int num_points = points.shape(0);
    for (int i = 0; i < num_points; ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();

    auto Bs           = vector<Array>();
    auto dB_by_dXs    = vector<Array>();
    auto d2B_by_dXdXs = vector<Array>();

    Bs.reserve(num_coils);
    dB_by_dXs.reserve(num_coils);
    d2B_by_dXdXs.reserve(num_coils);
    for(int i=0; i<num_coils; i++) {
        Bs.push_back(xt::zeros<double>({num_points, 3}));
        dB_by_dXs.push_back(xt::zeros<double>({num_points, 3, 3}));
        d2B_by_dXdXs.push_back(xt::zeros<double>({num_points, 3, 3, 3}));
    }

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        biot_savart_all_simd<Array>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], Bs[i], dB_by_dXs[i], d2B_by_dXdXs[i]);
        //biot_savart_B(points, gammas[i], dgamma_by_dphis[i], Bs[i]);
        //biot_savart_dB_by_dX(points, gammas[i], dgamma_by_dphis[i], dB_by_dXs[i]);
        //biot_savart_d2B_by_dXdX(points, gammas[i], dgamma_by_dphis[i], d2B_by_dXdXs[i]);
    }

    for(int i=0; i<num_coils; i++) {
        double fak1 = (currents[i] * 1e-7/gammas[i].shape(0));
        double fak2 = (1e-7/gammas[i].shape(0));
        for (int j1 = 0; j1 < num_points; ++j1) {
            for (int j2 = 0; j2 < 3; ++j2) {
                B(j1, j2) += fak1 * Bs[i](j1, j2);
                dB_by_coilcurrents[i](j1, j2) += fak2 * Bs[i](j1, j2);
                for (int j3 = 0; j3 < 3; ++j3) {
                    dB_by_dX(j1, j2, j3) += fak1 * dB_by_dXs[i](j1, j2, j3);
                    d2B_by_dXdcoilcurrents[i](j1, j2, j3) += fak2 * dB_by_dXs[i](j1, j2, j3);
                    for (int j4 = 0; j4 < 3; ++j4) {
                        d2B_by_dXdX(j1, j2, j3, j4) += fak1 * d2B_by_dXdXs[i](j1, j2, j3, j4);
                    }
                }
            }
        }
    }
}


template<class T>
void biot_savart_B_only_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& B) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto B_i   = Vec3dSimd();
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j          = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));

            auto diff = point_i - gamma_j;
            auto norm_diff_2     = normsq(diff);
            auto norm_diff       = sqrt(norm_diff_2);

            auto norm_diff_3_inv = 1./(norm_diff_2 * norm_diff);
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            B_i += dgamma_by_dphi_j_cross_diff * norm_diff_3_inv;
        }

        for(int j=0; j<simd_size; j++){
            B(i+j, 0) = B_i.x[j];
            B(i+j, 1) = B_i.y[j];
            B(i+j, 2) = B_i.z[j];
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        B(i, 0) = 0;
        B(i, 1) = 0;
        B(i, 2) = 0;
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point - gamma_j;
            double norm_diff = norm(diff);
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            auto B_i = dgamma_by_dphi_j_cross_diff / (norm_diff * norm_diff * norm_diff);

            B(i, 0) += B_i[0];
            B(i, 1) += B_i[1];
            B(i, 2) += B_i[2];
        }
    }
}
template void biot_savart_B_only_simd<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

void biot_savart_B_only(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& B) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    int num_points = points.shape(0);
    for (int i = 0; i < num_points; ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();

    auto Bs           = vector<Array>();

    Bs.reserve(num_coils);
    for(int i=0; i<num_coils; i++) {
        Bs.push_back(xt::zeros<double>({num_points, 3}));
    }

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        biot_savart_B_only_simd(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], Bs[i]);
    }

    for(int i=0; i<num_coils; i++) {
        double fak1 = (currents[i] * 1e-7/gammas[i].shape(0));
        for (int j1 = 0; j1 < num_points; ++j1) {
            for (int j2 = 0; j2 < 3; ++j2) {
                B(j1, j2) += fak1 * Bs[i](j1, j2);
            }
        }
    }
}


