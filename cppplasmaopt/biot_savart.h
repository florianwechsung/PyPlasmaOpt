#pragma once

#include "xtensor/xio.hpp"
#include "blaze/Blaze.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <tuple>


#define BLOCK_SIZE 4
typedef blaze::StaticVector<double,3UL> Vec3d;
typedef xt::pyarray<double> Array;
using blaze::DynamicMatrix;
using blaze::rowMajor;
using blaze::columnMajor;
typedef DynamicMatrix<double,rowMajor> RowMat;
typedef DynamicMatrix<double,columnMajor> ColMat;

Array biot_savart_B(Array& points, Array& gamma, Array& dgamma_by_dphi);
Array biot_savart_dB_by_dX(Array& points, Array& gamma, Array& dgamma_by_dphi);
Array biot_savart_dB_by_dcoilcoeff(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff);
Array biot_savart_dB_by_dcoilcoeff_via_chainrule(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff);
Array biot_savart_d2B_by_dXdcoilcoeff(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff);
Array biot_savart_d2B_by_dXdX(Array& points, Array& gamma, Array& dgamma_by_dphi);
Array biot_savart_d2B_by_dXdcoilcoeff_via_chainrule(Array& points, Array& gamma, Array& dgamma_by_dphi, Array& dgamma_by_dcoeff, Array& d2gamma_by_dphidcoeff);
