// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_VECTOR_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_VECTOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"

#include <NOX_Abstract_Vector.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class Vector : public ::NOX::Abstract::Vector
    {
     public:
      enum class MemoryType
      {
        View,
        Copy
      };

      //! Ctor that gets a shared_ptr with the state vector
      Vector(const std::shared_ptr<Core::LinAlg::Vector<double>>& source,
          MemoryType memory = MemoryType::Copy, ::NOX::CopyType type = ::NOX::DeepCopy);

      //! Ctor that captures the ownership of the vector
      Vector(Core::LinAlg::Vector<double>&& source);

      //! Ctor that copies the provided state vector or its shape
      Vector(const Core::LinAlg::Vector<double>& source, ::NOX::CopyType type = ::NOX::DeepCopy);

      //! Ctor that copies data or just shape from another NOX::Nln::Vector
      Vector(const NOX::Nln::Vector& source, ::NOX::CopyType type = ::NOX::DeepCopy);

      //! Main copy assignment operator
      ::NOX::Abstract::Vector& operator=(const NOX::Nln::Vector& source);

      //! Overloaded copy assignment operators
      ::NOX::Abstract::Vector& operator=(const Core::LinAlg::Vector<double>& source);
      ::NOX::Abstract::Vector& operator=(const ::NOX::Abstract::Vector& source) override;

      //! Clone method
      Teuchos::RCP<::NOX::Abstract::Vector> clone(
          ::NOX::CopyType type = ::NOX::DeepCopy) const override;

      //! Get reference to underlying Core::LinAlg::Vector.
      Core::LinAlg::Vector<double>& get_linalg_vector();

      //! Get const reference to underlying Core::LinAlg::Vector.
      const Core::LinAlg::Vector<double>& get_linalg_vector() const;

      //! Initialize every element of this vector with @p gamma .
      ::NOX::Abstract::Vector& init(double gamma) override;

      //! Initialize each element of this vector with a random value.
      ::NOX::Abstract::Vector& random(bool, int) override;

      //! Put element-wise absolute values of source vector @p y into this vector.
      ::NOX::Abstract::Vector& abs(const ::NOX::Abstract::Vector& y) override;

      //! Put element-wise reciprocal of source vector @p y into this vector.
      ::NOX::Abstract::Vector& reciprocal(const ::NOX::Abstract::Vector& y) override;

      //! Scale each element of this vector by @p gamma .
      ::NOX::Abstract::Vector& scale(double gamma) override;

      //! Scale this vector element-by-element by the vector @p a .
      ::NOX::Abstract::Vector& scale(const ::NOX::Abstract::Vector& a) override;

      //! Compute x = (alpha * a) + (gamma * x) where x is this vector.
      ::NOX::Abstract::Vector& update(
          double alpha, const ::NOX::Abstract::Vector& a, double gamma = 0.0) override;

      //! Compute x = (alpha * a) + (beta * b) + (gamma * x) where x is this vector.
      ::NOX::Abstract::Vector& update(double alpha, const ::NOX::Abstract::Vector& a, double beta,
          const ::NOX::Abstract::Vector& b, double gamma = 0.0) override;

      //! Return the vector norm.
      double norm(
          ::NOX::Abstract::Vector::NormType type = ::NOX::Abstract::Vector::TwoNorm) const override;

      //! Return the vector's weighted 2-norm.
      double norm(const ::NOX::Abstract::Vector&) const override;

      //! Return the inner product of the vector with @p y.
      double innerProduct(const ::NOX::Abstract::Vector& y) const override;

      //! Return the length of vector.
      ::NOX::size_type length() const override;

     private:
      //! Pointer to a storage vector owned by this object
      std::shared_ptr<Core::LinAlg::Vector<double>> linalg_vec_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
