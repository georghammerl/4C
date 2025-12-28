// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_REQUIRED_BASE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_REQUIRED_BASE_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Interface
    {
      class RequiredBase
      {
       public:
        enum FillType
        {
          //! The exact residual (F) is being calculated.
          Residual,
          //! The Jacobian matrix is being estimated.
          Jac,
          //! The preconditioner matrix is being estimated.
          Prec,
          //! The fill context is from a finite difference approximation
          FD_Res,
          //! The fill context is from a matrix free approximation
          MF_Res,
          //! The fill context is from a matrix free computeJacobian() approximation
          MF_Jac,
          //! A user defined estimation is being performed.
          User
        };

        RequiredBase() = default;

        //! Destructor
        virtual ~RequiredBase() = default;

        //! Compute the function, f, given the specified input vector x.  Returns true if
        //! computation was successful.
        virtual bool compute_f(const Core::LinAlg::Vector<double>& x,
            Core::LinAlg::Vector<double>& f, FillType fill_flag) = 0;
      };
    }  // namespace Interface
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
