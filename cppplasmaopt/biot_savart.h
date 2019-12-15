#pragma once

#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0

#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "blaze/Blaze.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <tuple>


typedef blaze::StaticVector<double,3UL> Vec3d;
typedef blaze::DynamicMatrix<double, blaze::rowMajor> RowMat;
typedef blaze::DynamicMatrix<double, blaze::columnMajor> ColMat;

typedef xt::pyarray<double> Array;


#include <vector>
using std::vector;

#include "xsimd/xsimd.hpp"
namespace xs = xsimd;
using xs::sqrt;
using vector_type = std::vector<double, xs::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;
using simd_t = xs::simd_type<double>;

struct Vec3dSimd {
    simd_t x;
    simd_t y;
    simd_t z;

    Vec3dSimd(const simd_t& x_, const simd_t& y_, const simd_t& z_) : x(x_), y(y_), z(z_) {
    }

    Vec3dSimd(double* xptr, double* yptr, double *zptr){
        x = xs::load_aligned(xptr);
        y = xs::load_aligned(yptr);
        z = xs::load_aligned(zptr);
    }

    void store_aligned(double* xptr, double* yptr, double *zptr){
        x.store_aligned(xptr);
        y.store_aligned(yptr);
        z.store_aligned(zptr);
    }

    simd_t& operator[] (int i){
        if(i==0) {
            return x;
        }else if(i==1){
            return y;
        } else{
            return z;
        }
    }

    friend Vec3dSimd operator+(Vec3dSimd lhs, const Vec3d& rhs) {
        lhs.x += rhs[0];
        lhs.y += rhs[1];
        lhs.z += rhs[2];
        return lhs;
    }

    Vec3dSimd& operator+=(const Vec3dSimd& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }

    friend Vec3dSimd operator-(Vec3dSimd lhs, const Vec3d& rhs) {
        lhs.x -= rhs[0];
        lhs.y -= rhs[1];
        lhs.z -= rhs[2];
        return lhs;
    }

    friend Vec3dSimd operator*(Vec3dSimd lhs, const simd_t& rhs) {
        lhs.x *= rhs;
        lhs.y *= rhs;
        lhs.z *= rhs;
        return lhs;
    }
};

vector<Array> biot_savart_B_allcoils(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis);
Array         biot_savart_dB_by_dX(Array& points, Array& gamma, Array& dgamma_by_dphi);
Array         biot_savart_d2B_by_dXdX(Array& points, Array& gamma, Array& dgamma_by_dphi);
vector<Array> biot_savart_dB_by_dcoilcoeff_via_chainrule_allcoils(vector<Array>& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs);
vector<Array> biot_savart_d2B_by_dXdcoilcoeff_via_chainrule_allcoils(vector<Array>& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs);
Array         biot_savart_dB_by_dcoilcoeff(Array& points, Array& gammas, Array& dgamma_by_dphis, Array& dgamma_by_dcoeffs, Array& d2gamma_by_dphidcoeffs);
Array         biot_savart_d2B_by_dXdcoilcoeff(Array& points, Array& gammas, Array& dgamma_by_dphis, Array& dgamma_by_dcoeffs, Array& d2gamma_by_dphidcoeffs);

template<class T>
void biot_savart_B_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& res);
