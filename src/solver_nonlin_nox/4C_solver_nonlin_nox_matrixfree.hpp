// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_MATRIXFREE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_MATRIXFREE_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian_base.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Epetra_Operator.h>
#include <Epetra_Vector.h>
#include <NOX_Epetra_MatrixFree.H>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class MatrixFree : public NOX::Nln::Interface::JacobianBase
    {
      class SparseOperatorWrapper : public Core::LinAlg::SparseOperator
      {
       public:
        SparseOperatorWrapper(Epetra_Operator& op);

        // Methods of Core::LinAlg::SparseOperator interface
        Epetra_Operator& epetra_operator() override;

        void zero() override;

        void reset() override;

        void assemble(int eid, const std::vector<int>& lmstride,
            const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
            const std::vector<int>& lmrowowner, const std::vector<int>& lmcol) override;

        void assemble(double val, int rgid, int cgid) override;

        bool filled() const override;

        void complete(Core::LinAlg::OptionsMatrixComplete options_matrix_complete = {}) override;

        void complete(const Core::LinAlg::Map& domainmap, const Core::LinAlg::Map& rangemap,
            Core::LinAlg::OptionsMatrixComplete options_matrix_complete = {}) override;

        void un_complete() override;

        void apply_dirichlet(
            const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock = true) override;

        void apply_dirichlet(const Core::LinAlg::Map& dbcmap, bool diagonalblock = true) override;


        const Core::LinAlg::Map& domain_map() const override;

        void add(const Core::LinAlg::SparseOperator& A, const bool transposeA, const double scalarA,
            const double scalarB) override;

        void add_other(Core::LinAlg::SparseMatrix& A, const bool transposeA, const double scalarA,
            const double scalarB) const override;

        void add_other(Core::LinAlg::BlockSparseMatrixBase& A, const bool transposeA,
            const double scalarA, const double scalarB) const override;

        int scale(double ScalarConstant) override;

        int multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
            Core::LinAlg::MultiVector<double>& Y) const override;

        // Methods of Epetra_Operator interface
        int SetUseTranspose(bool UseTranspose) override;

        int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

        int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

        double NormInf() const override;

        const char* Label() const override;

        bool UseTranspose() const override;

        bool HasNormInf() const override;

        const Epetra_Comm& Comm() const override;

        const Epetra_Map& OperatorDomainMap() const override;

        const Epetra_Map& OperatorRangeMap() const override;

       private:
        Epetra_Operator& operator_;
      };

     public:
      MatrixFree(Teuchos::ParameterList& printParams,
          const Teuchos::RCP<NOX::Nln::Interface::RequiredBase>& required,
          const NOX::Nln::Vector& cloneVector, double lambda, bool useNewPerturbation = false);

      Core::LinAlg::SparseOperator& get_operator();

      bool computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac) override;

     private:
      ::NOX::Epetra::MatrixFree matrix_free_;
      SparseOperatorWrapper wrapper_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
