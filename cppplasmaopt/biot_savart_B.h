#include "xtensor/xio.hpp"
#include "blaze/Blaze.h"

#define BLOCK_SIZE 4
typedef blaze::StaticVector<double,3UL> Vec3d;
typedef xt::pyarray<double> Array;

Array biot_savart_B(Array& points, Array& gamma, Array& dgamma_by_dphi) {
    int num_points = points.shape(0);
    Array res = xt::zeros<double>({num_points, 3});
    int num_quad_points = gamma.shape(0);
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        auto point = Vec3d(3, &points.at(i, 0));
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d(3, &gamma.at(j, 0));
            auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi.at(j, 0));
            auto diff = point - gamma_j;
            auto temp = cross(dgamma_by_dphi_j, diff) / std::pow(norm(diff), 3);
            res.at(i, 0) += temp[0];
            res.at(i, 1) += temp[1];
            res.at(i, 2) += temp[2];
        }
    }
    return res;
}

// attempt at blocking, but didn't seem to do much.
//Array biot_savart_B(Array& points, Array& gamma, Array& dgamma_by_dphi) {
    //int num_points = points.shape(0);
    //Array res = xt::zeros<double>({num_points, 3});
    //int num_quad_points = gamma.shape(0);
    //for (int ii = 0; ii < num_points; ii += BLOCK_SIZE) {
        //for (int jj = 0; jj < num_quad_points; jj += BLOCK_SIZE) {
            //int M = std::min(BLOCK_SIZE, num_points-ii);
            //int N = std::min(BLOCK_SIZE, num_quad_points-jj);

            //for (int i = ii; i < ii+M; ++i) {
                //auto point = Vec3d(3, &points.at(i, 0));
                //for (int j = jj; j < jj+N; ++j) {
                    //auto gamma_j = Vec3d(3, &gamma.at(j, 0));
                    //auto dgamma_by_dphi_j = Vec3d(3, &dgamma_by_dphi.at(j, 0));
                    //auto diff = point - gamma_j;
                    //auto temp = cross(dgamma_by_dphi_j, diff) / std::pow(norm(diff), 3);
                    //res.at(i, 0) += temp[0];
                    //res.at(i, 1) += temp[1];
                    //res.at(i, 2) += temp[2];
                //}
            //}
        //}
    //}
    //return res;
//}
