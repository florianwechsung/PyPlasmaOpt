#include "biot_savart.h"

template<class T>
void biot_savart_by_dcoilcoeff_all_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& dgamma_by_dcoeff, T& d2gamma_by_dphidcoeff, T& dB_by_dcoilcoeff, T& d2B_by_dXdcoilcoeff) {
    int num_points      = pointsx.size();
    int num_coil_coeffs = dgamma_by_dcoeff.shape(1);
    int num_quad_points = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        for(int k=0; k<num_coil_coeffs; k++) {
            auto B_i   = Vec3dSimd();
            auto dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
            for (int j = 0; j < num_quad_points; ++j) {
                auto gamma_j = Vec3d(3, &gamma(j, 0));
                auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi(j, 0));
                Vec3d three_dgamma_j_by_dphi = 3*dgamma_j_by_dphi;
                auto diff = point_i - gamma_j;
                auto norm_diff_2 = normsq(diff);
                auto norm_diff = sqrt(norm_diff_2);
                auto norm_diff_3_inv = 1/(norm_diff_2*norm_diff);
                auto norm_diff_5_inv = norm_diff_3_inv/(norm_diff_2);
                auto three_dgamma_by_dphi_cross_diff = cross(three_dgamma_j_by_dphi, diff);

                auto d2gamma_j_by_dphi_dcoeff_k = Vec3d(3, &d2gamma_by_dphidcoeff(j, k, 0));
                auto dgamma_j_by_dcoeff_k = Vec3d(3, &dgamma_by_dcoeff(j, k, 0));

                auto term1 = cross(d2gamma_j_by_dphi_dcoeff_k, diff) * norm_diff_3_inv;

                auto dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k = cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);
                auto term2 = Vec3dSimd(dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[0] * norm_diff_3_inv,
                        dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[1] * norm_diff_3_inv,
                        dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[2] * norm_diff_3_inv);

                auto diff_inner_dgamma_j_by_dcoeff_k = inner(dgamma_j_by_dcoeff_k, diff);
                auto term3 = three_dgamma_by_dphi_cross_diff * diff_inner_dgamma_j_by_dcoeff_k * norm_diff_5_inv ;
                auto temp = term1 - term2 + term3;
                B_i += temp;

                auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff_2);
                auto d2gamma_j_by_dphi_dcoeff_k_cross_diff       = cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                for (int l = 0; l < 3; ++l) {
                    auto el  = Vec3d{0., 0., 0.};
                    el[l] += 1.;

                    auto d2gamma_j_by_dphi_dcoeff_k_cross_el = cross(d2gamma_j_by_dphi_dcoeff_k, el);
                    auto dgamma_j_by_dphi_cross_el = cross(dgamma_j_by_dphi, el);
                    auto term1 = Vec3dSimd(d2gamma_j_by_dphi_dcoeff_k_cross_el[0], d2gamma_j_by_dphi_dcoeff_k_cross_el[1], d2gamma_j_by_dphi_dcoeff_k_cross_el[2]) * norm_diff_3_inv;
                    auto term2 = Vec3dSimd(dgamma_j_by_dphi_cross_el[0], dgamma_j_by_dphi_cross_el[1], dgamma_j_by_dphi_cross_el[2]) * (3. * diff_inner_dgamma_j_by_dcoeff_k * norm_diff_5_inv);
                    auto term3 = three_dgamma_by_dphi_cross_diff * (-5. * diff_inner_dgamma_j_by_dcoeff_k * diff[l] * norm_diff_7_inv);
                    auto term4 = three_dgamma_by_dphi_cross_diff * (dgamma_j_by_dcoeff_k[l] * norm_diff_5_inv);
                    auto term5 = d2gamma_j_by_dphi_dcoeff_k_cross_diff * (-3. * diff[l] * norm_diff_5_inv);
                    auto term6_fak = 3. * diff[l] * norm_diff_5_inv;
                    auto term6 = Vec3dSimd(dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[0] * term6_fak,
                            dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[1] * term6_fak,
                            dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[2] * term6_fak);
                    auto temp = term1 + term2 + term3 + term4 + term5 + term6;
                    dB_dX_i[l] += temp;
                }
            }
            for(int j=0; j<simd_size; j++){
                dB_by_dcoilcoeff(i+j, k, 0) = B_i.x[j];
                dB_by_dcoilcoeff(i+j, k, 1) = B_i.y[j];
                dB_by_dcoilcoeff(i+j, k, 2) = B_i.z[j];
                for (int l = 0; l < 3; ++l) {
                    d2B_by_dXdcoilcoeff(i+j, k, l, 0) += dB_dX_i[l].x[j];
                    d2B_by_dXdcoilcoeff(i+j, k, l, 1) += dB_dX_i[l].y[j];
                    d2B_by_dXdcoilcoeff(i+j, k, l, 2) += dB_dX_i[l].z[j];
                }
            }
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point - gamma_j;
            auto norm_diff = norm(diff);
            auto norm_diff_3_inv = 1/(norm_diff*norm_diff*norm_diff);
            auto norm_diff_5_inv = norm_diff_3_inv/(norm_diff*norm_diff);
            auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff*norm_diff);
            auto dgamma_by_dphi_cross_diff = cross(dgamma_j_by_dphi, diff);
            auto three_dgamma_by_dphi_cross_diff = dgamma_by_dphi_cross_diff * 3;

            for(int k=0; k<num_coil_coeffs; k++) {
                auto d2gamma_j_by_dphi_dcoeff_k = Vec3d(3, &d2gamma_by_dphidcoeff(j, k, 0));
                auto dgamma_j_by_dcoeff_k = Vec3d(3, &dgamma_by_dcoeff(j, k, 0));

                auto diff_inner_dgamma_j_by_dcoeff_k              = inner(diff, dgamma_j_by_dcoeff_k);
                Vec3d d2gamma_j_by_dphi_dcoeff_k_cross_diff       = cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                Vec3d dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k = cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);

                auto term1 = norm_diff_3_inv * cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                auto term2 = norm_diff_3_inv * cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);
                auto term3 = norm_diff_5_inv * diff_inner_dgamma_j_by_dcoeff_k * three_dgamma_by_dphi_cross_diff;
                auto temp = term1 - term2 + term3;
                dB_by_dcoilcoeff(i, k, 0) += temp[0];
                dB_by_dcoilcoeff(i, k, 1) += temp[1];
                dB_by_dcoilcoeff(i, k, 2) += temp[2];
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
                    d2B_by_dXdcoilcoeff(i, k, l, 0) += temp[0];
                    d2B_by_dXdcoilcoeff(i, k, l, 1) += temp[1];
                    d2B_by_dXdcoilcoeff(i, k, l, 2) += temp[2];
                }
            }
        }
    }
}
//template void biot_savart_all_simd<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);


void biot_savart_by_dcoilcoeff_all(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs, vector<double>& currents, vector<Array>& dB_by_dcoilcoeffs, vector<Array>& d2B_by_dXdcoilcoeff) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        biot_savart_by_dcoilcoeff_all_simd<Array>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],dgamma_by_dcoeffs[i], d2gamma_by_dphidcoeffs[i], dB_by_dcoilcoeffs[i], d2B_by_dXdcoilcoeff[i]);
        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        dB_by_dcoilcoeffs[i] *= fak;
        d2B_by_dXdcoilcoeff[i] *= fak;
    }
}

template<class T>
void biot_savart_B_only_vjp_impl(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto v_i   = Vec3dSimd();
        auto vgrad_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
        for(int k=0; k<simd_size; k++){
            for (int d = 0; d < 3; ++d) {
                v_i[d][k] = v(i+k, d);
                for (int dd = 0; dd < 3; ++dd) {
                    vgrad_i[dd][d][k] = vgrad(i+k, dd, d);
                }
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi(j, 0));
            Vec3d three_dgamma_j_by_dphi = 3*dgamma_j_by_dphi;
            auto diff = point_i - gamma_j;
            auto norm_diff_2 = normsq(diff);
            auto norm_diff = sqrt(norm_diff_2);
            auto norm_diff_3_inv = 1/(norm_diff_2*norm_diff);
            auto norm_diff_5_inv = norm_diff_3_inv/(norm_diff_2);
            auto three_dgamma_by_dphi_cross_diff = cross(three_dgamma_j_by_dphi, diff);
            for(int d=0; d<3; d++){
                auto ed = Vec3d();
                ed[d] += 1.0;

                auto d2gamma_j_by_dphi_dcoeff_k = ed;
                auto dgamma_j_by_dcoeff_k = ed;

                auto term1 = cross(d2gamma_j_by_dphi_dcoeff_k, diff) * norm_diff_3_inv;

                auto dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k = cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);
                auto term2 = Vec3dSimd(dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[0] * norm_diff_3_inv,
                        dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[1] * norm_diff_3_inv,
                        dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[2] * norm_diff_3_inv);

                auto diff_inner_dgamma_j_by_dcoeff_k = inner(dgamma_j_by_dcoeff_k, diff);
                auto term3 = three_dgamma_by_dphi_cross_diff * diff_inner_dgamma_j_by_dcoeff_k * norm_diff_5_inv ;

                auto res_dgamma_by_dphi_add = inner(term1, v_i);
                auto res_gamma_add = inner(term3, v_i)-inner(term2, v_i);

                for(int k=0; k<simd_size; k++){
                    res_gamma(j, d) += res_gamma_add[k];
                    res_dgamma_by_dphi(j, d) += res_dgamma_by_dphi_add[k];
                }

                auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff_2);
                auto d2gamma_j_by_dphi_dcoeff_k_cross_diff       = cross(ed, diff);
                for (int l = 0; l < 3; ++l) {
                    auto el  = Vec3d{0., 0., 0.};
                    el[l] += 1.;

                    auto d2gamma_j_by_dphi_dcoeff_k_cross_el = cross(d2gamma_j_by_dphi_dcoeff_k, el);
                    auto dgamma_j_by_dphi_cross_el = cross(dgamma_j_by_dphi, el);
                    auto term1 = Vec3dSimd(d2gamma_j_by_dphi_dcoeff_k_cross_el[0], d2gamma_j_by_dphi_dcoeff_k_cross_el[1], d2gamma_j_by_dphi_dcoeff_k_cross_el[2]) * norm_diff_3_inv;
                    auto term2 = Vec3dSimd(dgamma_j_by_dphi_cross_el[0], dgamma_j_by_dphi_cross_el[1], dgamma_j_by_dphi_cross_el[2]) * (3. * diff_inner_dgamma_j_by_dcoeff_k * norm_diff_5_inv);
                    auto term3 = three_dgamma_by_dphi_cross_diff * (-5. * diff_inner_dgamma_j_by_dcoeff_k * diff[l] * norm_diff_7_inv);
                    auto term4 = three_dgamma_by_dphi_cross_diff * (dgamma_j_by_dcoeff_k[l] * norm_diff_5_inv);
                    auto term5 = d2gamma_j_by_dphi_dcoeff_k_cross_diff * (-3. * diff[l] * norm_diff_5_inv);
                    auto term6_fak = 3. * diff[l] * norm_diff_5_inv;
                    auto term6 = Vec3dSimd(dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[0] * term6_fak,
                            dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[1] * term6_fak,
                            dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k[2] * term6_fak);

                    auto temp1 = term2 + term3 + term4 + term6;
                    auto temp2 = term1 + term5;
                    
                    auto temp1vi = inner(temp1, vgrad_i[l]);
                    auto temp2vi = inner(temp2, vgrad_i[l]);
                    for(int k=0; k<simd_size; k++){
                        res_gamma(j, d) += temp1vi[k];
                        res_dgamma_by_dphi(j, d) += temp2vi[k];
                    }
                }
            }
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        auto v_i   = Vec3d();
        auto vgrad_i = vector<Vec3d>{
                Vec3d(), Vec3d(), Vec3d()
            };
        for (int d = 0; d < 3; ++d) {
            v_i[d] = v(i, d);
            for (int dd = 0; dd < 3; ++dd) {
                vgrad_i[dd][d] = vgrad(i, dd, d);
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma(j, 0));
            auto dgamma_j_by_dphi = Vec3d(3, &dgamma_by_dphi(j, 0));
            auto diff = point - gamma_j;
            auto norm_diff = norm(diff);
            auto norm_diff_3_inv = 1/(norm_diff*norm_diff*norm_diff);
            auto norm_diff_5_inv = norm_diff_3_inv/(norm_diff*norm_diff);
            auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff*norm_diff);
            auto dgamma_by_dphi_cross_diff = cross(dgamma_j_by_dphi, diff);
            auto three_dgamma_by_dphi_cross_diff = dgamma_by_dphi_cross_diff * 3;

            for(int d=0; d<3; d++){
                auto ed = Vec3d();
                ed[d] += 1.0;
                auto d2gamma_j_by_dphi_dcoeff_k = ed;
                auto dgamma_j_by_dcoeff_k = ed;

                auto diff_inner_dgamma_j_by_dcoeff_k              = inner(diff, dgamma_j_by_dcoeff_k);
                Vec3d d2gamma_j_by_dphi_dcoeff_k_cross_diff       = cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                Vec3d dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k = cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);

                auto term1 = norm_diff_3_inv * cross(d2gamma_j_by_dphi_dcoeff_k, diff);
                auto res_dgamma_by_dphi_add = inner(term1, v_i);

                auto term2 = norm_diff_3_inv * cross(dgamma_j_by_dphi, dgamma_j_by_dcoeff_k);
                auto term3 = norm_diff_5_inv * diff_inner_dgamma_j_by_dcoeff_k * three_dgamma_by_dphi_cross_diff;
                auto res_gamma_add = inner(term3, v_i)-inner(term2, v_i);

                res_gamma(j, d) += res_gamma_add;
                res_dgamma_by_dphi(j, d) += res_dgamma_by_dphi_add;

                for (int l = 0; l < 3; ++l) {
                    auto el  = Vec3d{0., 0., 0.};
                    el[l] = 1.0;
                    auto term1 = norm_diff_3_inv * cross(d2gamma_j_by_dphi_dcoeff_k, el);
                    auto term2 = 3 * diff_inner_dgamma_j_by_dcoeff_k * norm_diff_5_inv * cross(dgamma_j_by_dphi, el);
                    auto term3 = -15 * diff_inner_dgamma_j_by_dcoeff_k * diff[l] * norm_diff_7_inv * dgamma_by_dphi_cross_diff;
                    auto term4 = 3 * dgamma_j_by_dcoeff_k[l] * norm_diff_5_inv * dgamma_by_dphi_cross_diff;
                    auto term5 = -3 * diff[l] * norm_diff_5_inv * d2gamma_j_by_dphi_dcoeff_k_cross_diff;
                    auto term6 = 3 * diff[l] * norm_diff_5_inv * dgamma_j_by_dphi_cross_dgamma_j_by_dcoeff_k;

                    auto temp1 = term2 + term3 + term4 + term6;
                    auto temp2 = term1 + term5;
                    
                    auto temp1vi = inner(temp1, vgrad_i[l]);
                    auto temp2vi = inner(temp2, vgrad_i[l]);
                    res_gamma(j, d) += temp1vi;
                    res_dgamma_by_dphi(j, d) += temp2vi;
                }
            }
        }
    }
}

template void biot_savart_B_only_vjp_impl<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

void biot_savart_by_dcoilcoeff_all_vjp(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, vector<Array>& res_gamma, vector<Array>& res_dgamma_by_dphi, Array& vgrad, vector<Array>& res_grad_gamma, vector<Array>& res_grad_dgamma_by_dphi) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        biot_savart_B_only_vjp_impl<Array>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                v, res_gamma[i], res_dgamma_by_dphi[i],
                vgrad, res_grad_gamma[i], res_grad_dgamma_by_dphi[i]);
        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        res_gamma[i] *= fak;
        res_dgamma_by_dphi[i] *= fak;
        res_grad_gamma[i] *= fak;
        res_grad_dgamma_by_dphi[i] *= fak;
    }
}
