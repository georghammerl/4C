// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_MATRIXFREE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_MATRIXFREE_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Epetra_Operator.h>
#include <Epetra_Vector.h>
#include <NOX_Epetra_Interface_Jacobian.H>
#include <NOX_Epetra_MatrixFree.H>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class MatrixFree : public ::NOX::Epetra::Interface::Jacobian
    {
     public:
      MatrixFree(Teuchos::ParameterList& printParams,
          const Teuchos::RCP<NOX::Nln::Interface::RequiredBase>& required,
          const NOX::Nln::Vector& cloneVector, double lambda, bool useNewPerturbation = false);

      ::NOX::Epetra::MatrixFree& get_matrix_free();

      bool computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac) override;

     private:
      ::NOX::Epetra::MatrixFree matrix_free_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
