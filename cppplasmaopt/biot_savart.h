#pragma once

#include "xtensor/xio.hpp"
#include "blaze/Blaze.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

#define BLOCK_SIZE 4
typedef blaze::StaticVector<double,3UL> Vec3d;
typedef xt::pyarray<double> Array;
Array biot_savart_B(Array& points, Array& gamma, Array& dgamma_by_dphi);
Array biot_savart_dB_by_dX(Array& points, Array& gamma, Array& dgamma_by_dphi);
