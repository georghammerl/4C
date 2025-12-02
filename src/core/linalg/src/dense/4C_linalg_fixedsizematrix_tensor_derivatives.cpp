// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"

#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_symmetric_einstein.hpp"



FOUR_C_NAMESPACE_OPEN

void Core::LinAlg::FourTensorOperations::add_derivative_of_squared_tensor(
    Core::LinAlg::Matrix<6, 6>& C, double scalar_squared_dx, Core::LinAlg::Matrix<3, 3> X,
    double scalar_this)
{
  C(0, 0) = scalar_this * C(0, 0) + scalar_squared_dx * 2. * X(0, 0);  // C1111
  C(0, 1) = scalar_this * C(0, 1);                                     // C1122
  C(0, 2) = scalar_this * C(0, 2);                                     // C1133
  C(0, 3) = scalar_this * C(0, 3) + scalar_squared_dx * X(0, 1);       // C1112
  C(0, 4) = scalar_this * C(0, 4);                                     // C1123
  C(0, 5) = scalar_this * C(0, 5) + scalar_squared_dx * X(0, 2);       // C1113

  C(1, 0) = scalar_this * C(1, 0);                                     // C2211
  C(1, 1) = scalar_this * C(1, 1) + scalar_squared_dx * 2. * X(1, 1);  // C2222
  C(1, 2) = scalar_this * C(1, 2);                                     // C2233
  C(1, 3) = scalar_this * C(1, 3) + scalar_squared_dx * X(0, 1);       // C2212
  C(1, 4) = scalar_this * C(1, 4) + scalar_squared_dx * X(1, 2);       // C2223
  C(1, 5) = scalar_this * C(1, 5);                                     // C2213

  C(2, 0) = scalar_this * C(2, 0);                                     // C3311
  C(2, 1) = scalar_this * C(2, 1);                                     // C3322
  C(2, 2) = scalar_this * C(2, 2) + scalar_squared_dx * 2. * X(2, 2);  // C3333
  C(2, 3) = scalar_this * C(2, 3);                                     // C3312
  C(2, 4) = scalar_this * C(2, 4) + scalar_squared_dx * X(1, 2);       // C3323
  C(2, 5) = scalar_this * C(2, 5) + scalar_squared_dx * X(0, 2);       // C3313

  C(3, 0) = scalar_this * C(3, 0) + scalar_squared_dx * X(0, 1);                    // C1211
  C(3, 1) = scalar_this * C(3, 1) + scalar_squared_dx * X(0, 1);                    // C1222
  C(3, 2) = scalar_this * C(3, 2);                                                  // C1233
  C(3, 3) = scalar_this * C(3, 3) + scalar_squared_dx * 0.5 * (X(0, 0) + X(1, 1));  // C1212
  C(3, 4) = scalar_this * C(3, 4) + scalar_squared_dx * 0.5 * X(0, 2);              // C1223
  C(3, 5) = scalar_this * C(3, 5) + scalar_squared_dx * 0.5 * X(1, 2);              // C1213

  C(4, 0) = scalar_this * C(4, 0);                                                  // C2311
  C(4, 1) = scalar_this * C(4, 1) + scalar_squared_dx * X(1, 2);                    // C2322
  C(4, 2) = scalar_this * C(4, 2) + scalar_squared_dx * X(1, 2);                    // C2333
  C(4, 3) = scalar_this * C(4, 3) + scalar_squared_dx * 0.5 * X(0, 2);              // C2312
  C(4, 4) = scalar_this * C(4, 4) + scalar_squared_dx * 0.5 * (X(1, 1) + X(2, 2));  // C2323
  C(4, 5) = scalar_this * C(4, 5) + scalar_squared_dx * 0.5 * X(0, 1);              // C2313

  C(5, 0) = scalar_this * C(5, 0) + scalar_squared_dx * X(0, 2);                    // C1311
  C(5, 1) = scalar_this * C(5, 1);                                                  // C1322
  C(5, 2) = scalar_this * C(5, 2) + scalar_squared_dx * X(0, 2);                    // C1333
  C(5, 3) = scalar_this * C(5, 3) + scalar_squared_dx * 0.5 * X(1, 2);              // C1312
  C(5, 4) = scalar_this * C(5, 4) + scalar_squared_dx * 0.5 * X(0, 1);              // C1323
  C(5, 5) = scalar_this * C(5, 5) + scalar_squared_dx * 0.5 * (X(2, 2) + X(0, 0));  // C1313
}

Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>
Core::LinAlg::FourTensorOperations::derivative_of_inva_b_inva_product(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& invA,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& invABinvA)
{
  return -0.5 *
         (einsum_sym<"ik", "jl">(invA, invABinvA) + einsum_sym<"il", "jk">(invA, invABinvA) +
             einsum_sym<"jk", "il">(invA, invABinvA) + einsum_sym<"jl", "ik">(invA, invABinvA));
}


FOUR_C_NAMESPACE_CLOSE
