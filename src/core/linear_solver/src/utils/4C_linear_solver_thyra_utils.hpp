// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_THYRA_UTILS_HPP
#define FOUR_C_LINEAR_SOLVER_THYRA_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <Thyra_DefaultBlockedLinearOp_decl.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver::Utils
{
  Teuchos::RCP<const Thyra::VectorSpaceBase<double>> create_thyra_map(const LinAlg::Map& map);

  Teuchos::RCP<Thyra::MultiVectorBase<double>> create_thyra_multi_vector(
      const LinAlg::MultiVector<double>& multi_vector, const LinAlg::Map& map);

  Teuchos::RCP<const Thyra::LinearOpBase<double>> create_thyra_linear_op(
      const LinAlg::SparseMatrix& matrix, LinAlg::DataAccess access);

  Teuchos::RCP<const Thyra::LinearOpBase<double>> create_thyra_linear_op(
      const LinAlg::BlockSparseMatrixBase& matrix, LinAlg::DataAccess access);
}  // namespace Core::LinearSolver::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
