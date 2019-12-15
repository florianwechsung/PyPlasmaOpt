#define FORCE_IMPORT_ARRAY
#include "xtensor/xnpy.hpp"
#include "biot_savart.h"
#include <chrono>




int main() {
    xt::xarray<double> points         = xt::load_npy<double>("points_200.npy");
    xt::xarray<double> gamma          = xt::load_npy<double>("gamma_200.npy");
    xt::xarray<double> dgamma_by_dphi = xt::load_npy<double>("dgamma_by_dphi_200.npy");
    int n = 10000;

    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        auto res = xt::xarray<double>::from_shape({points.shape(0), 3, 3});
        biot_savart_dB_by_dX_simd( pointsx,  pointsy,  pointsz, gamma, dgamma_by_dphi, res);
        if(i==0)
            std::cout << res(0, 0) << " " << res(40, 2) << std::endl;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        auto res = xt::xarray<double>::from_shape({points.shape(0), 3});
        biot_savart_B_simd( pointsx,  pointsy,  pointsz, gamma, dgamma_by_dphi, res);
        if(i==0)
            std::cout << res(0, 0) << " " << res(40, 2) << std::endl;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double dB_by_dX_simdtime = std::chrono::duration_cast<std::chrono::milliseconds>( t3 - t2 ).count();
    double dB_by_dX_gflops = 1e-9 * n * 100 * points.shape(0) * gamma.shape(0);
    std::cout << "dB_dX simd     " << dB_by_dX_simdtime << "ms, " << (dB_by_dX_gflops/(0.001 * dB_by_dX_simdtime)) << "GFlops" << std::endl;
    double B_simdtime = std::chrono::duration_cast<std::chrono::milliseconds>( t4 - t3 ).count();
    double B_gflops = 1e-9 * n * 30 * points.shape(0) * gamma.shape(0);
    std::cout << "B     simd     " << B_simdtime << "ms, " << (B_gflops/(0.001 * B_simdtime)) << "GFlops" << std::endl;
    return 1;
}
