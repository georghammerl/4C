// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_vector.hpp"

#include "4C_utils_exceptions.hpp"

#include <Epetra_Vector.h>

FOUR_C_NAMESPACE_OPEN

NOX::Nln::Vector::Vector(
    const Teuchos::RCP<Epetra_Vector>& source, MemoryType memoryType, ::NOX::CopyType type)
    : ::NOX::Epetra::Vector(source,
          memoryType == MemoryType::View ? ::NOX::Epetra::Vector::CreateView
                                         : ::NOX::Epetra::Vector::CreateCopy,
          type)
{
}

NOX::Nln::Vector::Vector(const Epetra_Vector& source, ::NOX::CopyType type)
    : ::NOX::Epetra::Vector(source, type)
{
}

NOX::Nln::Vector::Vector(const NOX::Nln::Vector& source, ::NOX::CopyType type)
    : ::NOX::Epetra::Vector(source, type)
{
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const Epetra_Vector& source)
{
  epetraVec->Scale(1.0, source);
  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const NOX::Nln::Vector& source)
{
  epetraVec->Scale(1.0, source.getEpetraVector());
  return *this;
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const ::NOX::Abstract::Vector& source)
{
  return operator=(dynamic_cast<const NOX::Nln::Vector&>(source));
}

::NOX::Abstract::Vector& NOX::Nln::Vector::operator=(const ::NOX::Epetra::Vector&)
{
  FOUR_C_THROW("This operator= should never be called!");

  return *this;
}

Teuchos::RCP<::NOX::Abstract::Vector> NOX::Nln::Vector::clone(::NOX::CopyType type) const
{
  Teuchos::RCP<::NOX::Abstract::Vector> newVec =
      Teuchos::rcp(new NOX::Nln::Vector(*epetraVec, type));
  return newVec;
}

FOUR_C_NAMESPACE_CLOSE
