// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_matrixfree.hpp"

FOUR_C_NAMESPACE_OPEN

NOX::Nln::MatrixFree::SparseOperatorWrapper::SparseOperatorWrapper(Epetra_Operator& op)
    : operator_(op)
{
}

Epetra_Operator& NOX::Nln::MatrixFree::SparseOperatorWrapper::epetra_operator()
{
  return operator_;
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::zero() { FOUR_C_THROW("Not implemented"); }

void NOX::Nln::MatrixFree::SparseOperatorWrapper::reset() { FOUR_C_THROW("Not implemented"); }

void NOX::Nln::MatrixFree::SparseOperatorWrapper::assemble(int eid,
    const std::vector<int>& lmstride, const Core::LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
    const std::vector<int>& lmcol)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::assemble(double val, int rgid, int cgid)
{
  FOUR_C_THROW("Not implemented");
}

bool NOX::Nln::MatrixFree::SparseOperatorWrapper::filled() const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::complete(
    Core::LinAlg::OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::complete(const Core::LinAlg::Map& domainmap,
    const Core::LinAlg::Map& rangemap, Core::LinAlg::OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::un_complete() { FOUR_C_THROW("Not implemented"); }

void NOX::Nln::MatrixFree::SparseOperatorWrapper::apply_dirichlet(
    const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::apply_dirichlet(
    const Core::LinAlg::Map& dbcmap, bool diagonalblock)
{
  FOUR_C_THROW("Not implemented");
}

bool NOX::Nln::MatrixFree::SparseOperatorWrapper::is_dbc_applied(const Core::LinAlg::Map& dbcmap,
    bool diagonalblock, const Core::LinAlg::SparseMatrix* trafor) const
{
  FOUR_C_THROW("Not implemented");
}

const Core::LinAlg::Map& NOX::Nln::MatrixFree::SparseOperatorWrapper::domain_map() const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::add(const Core::LinAlg::SparseOperator& A,
    const bool transposeA, const double scalarA, const double scalarB)
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::add_other(Core::LinAlg::SparseMatrix& A,
    const bool transposeA, const double scalarA, const double scalarB) const
{
  FOUR_C_THROW("Not implemented");
}

void NOX::Nln::MatrixFree::SparseOperatorWrapper::add_other(Core::LinAlg::BlockSparseMatrixBase& A,
    const bool transposeA, const double scalarA, const double scalarB) const
{
  FOUR_C_THROW("Not implemented");
}

int NOX::Nln::MatrixFree::SparseOperatorWrapper::scale(double ScalarConstant)
{
  FOUR_C_THROW("Not implemented");
}

int NOX::Nln::MatrixFree::SparseOperatorWrapper::multiply(bool TransA,
    const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y) const
{
  FOUR_C_ASSERT(!TransA, "Transposed multiplication is not supported");
  return operator_.Apply(X.get_epetra_multi_vector(), Y.get_epetra_multi_vector());
}


int NOX::Nln::MatrixFree::SparseOperatorWrapper::SetUseTranspose(bool UseTranspose)
{
  FOUR_C_THROW("Not implemented");
  return -1;
}


int NOX::Nln::MatrixFree::SparseOperatorWrapper::Apply(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  FOUR_C_THROW("Not implemented");
  return -1;
}


int NOX::Nln::MatrixFree::SparseOperatorWrapper::ApplyInverse(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  FOUR_C_THROW("Not implemented");
  return -1;
}


double NOX::Nln::MatrixFree::SparseOperatorWrapper::NormInf() const
{
  FOUR_C_THROW("Not implemented");
  return -1;
}


const char* NOX::Nln::MatrixFree::SparseOperatorWrapper::Label() const
{
  FOUR_C_THROW("Not implemented");
  return nullptr;
}


bool NOX::Nln::MatrixFree::SparseOperatorWrapper::UseTranspose() const
{
  FOUR_C_THROW("Not implemented");
  return false;
}


bool NOX::Nln::MatrixFree::SparseOperatorWrapper::HasNormInf() const
{
  FOUR_C_THROW("Not implemented");
  return false;
}


const Epetra_Comm& NOX::Nln::MatrixFree::SparseOperatorWrapper::Comm() const
{
  FOUR_C_THROW("Not implemented");
}


const Epetra_Map& NOX::Nln::MatrixFree::SparseOperatorWrapper::OperatorDomainMap() const
{
  FOUR_C_THROW("Not implemented");
}


const Epetra_Map& NOX::Nln::MatrixFree::SparseOperatorWrapper::OperatorRangeMap() const
{
  FOUR_C_THROW("Not implemented");
}

NOX::Nln::MatrixFree::MatrixFree(Teuchos::ParameterList& printParams,
    const Teuchos::RCP<NOX::Nln::Interface::RequiredBase>& required,
    const NOX::Nln::Vector& cloneVector, double lambda, bool useNewPerturbation)
    : matrix_free_(printParams, required, cloneVector, useNewPerturbation), wrapper_(matrix_free_)
{
  matrix_free_.setLambda(lambda);
}

Core::LinAlg::SparseOperator& NOX::Nln::MatrixFree::get_operator() { return wrapper_; }

bool NOX::Nln::MatrixFree::computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac)
{
  return matrix_free_.computeJacobian(x, Jac);
}

FOUR_C_NAMESPACE_CLOSE
