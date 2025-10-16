// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_VECTOR_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_VECTOR_HPP

#include "4C_config.hpp"

#include <NOX_Epetra_Vector.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class Vector : public ::NOX::Epetra::Vector
    {
     public:
      enum class MemoryType
      {
        View = ::NOX::Epetra::Vector::CreateView,
        Copy = ::NOX::Epetra::Vector::CreateCopy
      };

      //! Reproduce ctors of ::NOX::Epetra::Vector
      Vector(const Teuchos::RCP<Epetra_Vector>& source, MemoryType memoryType = MemoryType::Copy,
          ::NOX::CopyType type = ::NOX::DeepCopy);

      Vector(const Epetra_Vector& source, ::NOX::CopyType type = ::NOX::DeepCopy);

      Vector(const NOX::Nln::Vector& source, ::NOX::CopyType type = ::NOX::DeepCopy);

      //! Main copy assignment operator
      ::NOX::Abstract::Vector& operator=(const NOX::Nln::Vector& source);

      //! Overloaded copy assignment operators
      ::NOX::Abstract::Vector& operator=(const Epetra_Vector& source) override;
      ::NOX::Abstract::Vector& operator=(const ::NOX::Abstract::Vector& source) override;

      //! This operator is inherited and should not be ever called
      ::NOX::Abstract::Vector& operator=(const ::NOX::Epetra::Vector&) override;

      //! Clone method
      Teuchos::RCP<::NOX::Abstract::Vector> clone(
          ::NOX::CopyType type = ::NOX::DeepCopy) const override;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
