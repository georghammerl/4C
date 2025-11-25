// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_matrixfree.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::MatrixFree::MatrixFree(Teuchos::ParameterList& printParams,
    const Teuchos::RCP<NOX::Nln::Interface::RequiredBase>& required,
    const NOX::Nln::Vector& cloneVector, double lambda, bool useNewPerturbation)
    : matrix_free_(printParams, required, cloneVector, useNewPerturbation)
{
  matrix_free_.setLambda(lambda);
}

::NOX::Epetra::MatrixFree& NOX::Nln::MatrixFree::get_matrix_free() { return matrix_free_; }

bool NOX::Nln::MatrixFree::computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac)
{
  return matrix_free_.computeJacobian(x, Jac);
}

FOUR_C_NAMESPACE_CLOSE
