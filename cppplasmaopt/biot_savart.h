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

    Vec3dSimd() : x(0.), y(0.), z(0.){
    }

    Vec3dSimd(double x_, double y_, double z_) : x(x_), y(y_), z(z_){
    }

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

    friend Vec3dSimd operator+(Vec3dSimd lhs, const Vec3dSimd& rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
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

    friend Vec3dSimd operator-(Vec3dSimd lhs, const Vec3dSimd& rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        return lhs;
    }

    friend Vec3dSimd operator*(Vec3dSimd lhs, const simd_t& rhs) {
        lhs.x *= rhs;
        lhs.y *= rhs;
        lhs.z *= rhs;
        return lhs;
    }
};


inline simd_t inner(Vec3d& b, Vec3dSimd& a){
    return a.x*b[0]+a.y*b[1]+a.z*b[2];
}

inline simd_t inner(Vec3dSimd& a, Vec3d& b){
    return a.x*b[0]+a.y*b[1]+a.z*b[2];
}

inline Vec3dSimd cross(Vec3dSimd& a, Vec3d& b){
    return Vec3dSimd(a.y * b[2] - a.z * b[1], a.z * b[0] - a.x * b[2], a.x * b[1] - a.y * b[0]);
}

inline Vec3dSimd cross(Vec3d& a, Vec3dSimd& b){
    return Vec3dSimd(a[1] * b.z - a[2] * b.y, a[2] * b.x - a[0] * b.z, a[0] * b.y - a[1] * b.x);
}

inline simd_t normsq(Vec3dSimd& a){
    return a.x*a.x+a.y*a.y+a.z*a.z;
}

void biot_savart_all(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& B, Array& dB_by_dX, Array& d2B_by_dXdX, vector<Array>& dB_by_coilcurrents, vector<Array>& d2B_by_dXdcoilcurrents);
void biot_savart_B_only(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& B);

void biot_savart_by_dcoilcoeff_all(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs, vector<double>& currents, vector<Array>& dB_by_dcoilcoeffs, vector<Array>& d2B_by_dXdcoilcoeff);

void biot_savart_B(Array& points, Array& gammas, Array& dgamma_by_dphis, Array& res);
void biot_savart_dB_by_dX(Array& points, Array& gammas, Array& dgamma_by_dphis, Array& res);
void biot_savart_d2B_by_dXdX(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& res);

void biot_savart_dB_by_dcoilcoeff(Array& points, Array& gammas, Array& dgamma_by_dphis, Array& dgamma_by_dcoeffs, Array& d2gamma_by_dphidcoeffs, Array& res);
void biot_savart_d2B_by_dXdcoilcoeff(Array& points, Array& gammas, Array& dgamma_by_dphis, Array& dgamma_by_dcoeffs, Array& d2gamma_by_dphidcoeffs, Array& res);

template<class T>
void biot_savart_all_simd(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX);
