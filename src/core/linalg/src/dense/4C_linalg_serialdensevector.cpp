// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_serialdensevector.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  SerialDenseVector::SerialDenseVector(int length) : vec_(length) {}

  SerialDenseVector::SerialDenseVector(int length, bool zeroOut) : vec_(length, zeroOut) {}

  SerialDenseVector::SerialDenseVector(Teuchos::DataAccess cv, double* values, int length)
      : vec_(cv, values, length)
  {
  }

  SerialDenseVector::SerialDenseVector(const Base& source, Teuchos::ETransp trans) : vec_(source)
  {
    switch (trans)
    {
      case Teuchos::NO_TRANS:
      case Teuchos::TRANS:
      case Teuchos::CONJ_TRANS:
        break;
      default:
        FOUR_C_THROW("Unsupported transposition flag for vector copy constructor");
    }
  }

  int SerialDenseVector::length() const { return vec_.length(); }
  int SerialDenseVector::num_rows() const { return vec_.numRows(); }
  int SerialDenseVector::numRows() const { return vec_.numRows(); }
  double SerialDenseVector::normInf() const { return vec_.normInf(); }
  double SerialDenseVector::normOne() const { return vec_.normOne(); }
  double SerialDenseVector::norm2() const { return vec_.normFrobenius(); }
  double SerialDenseVector::normFrobenius() const { return vec_.normFrobenius(); }
  bool SerialDenseVector::empty() const { return vec_.length() == 0; }

  double* SerialDenseVector::values() const { return vec_.values(); }
  double* SerialDenseVector::data() const { return values(); }

  int SerialDenseVector::size(int length) { return vec_.size(length); }

  int SerialDenseVector::Size(int length) { return size(length); }

  int SerialDenseVector::resize(int length) { return vec_.resize(length); }

  int SerialDenseVector::scale(double alpha) { return vec_.scale(alpha); }

  int SerialDenseVector::put_scalar(double val) { return vec_.putScalar(val); }

  int SerialDenseVector::putScalar(double val) { return vec_.putScalar(val); }

  void SerialDenseVector::assign(const SerialDenseVector& source) { vec_.assign(source.base()); }

  SerialDenseVector& SerialDenseVector::operator+=(const SerialDenseVector& other)
  {
    vec_ += other.vec_;
    return *this;
  }

  double SerialDenseVector::dot(const SerialDenseVector& other) const
  {
    return vec_.dot(other.base());
  }

  int SerialDenseVector::multiply(Teuchos::ETransp transa, Teuchos::ETransp transb, double alpha,
      const Teuchos::SerialDenseMatrix<ordinalType, scalarType>& A,
      const Teuchos::SerialDenseMatrix<ordinalType, scalarType>& B, double beta)
  {
    return vec_.multiply(transa, transb, alpha, A, B, beta);
  }

  int SerialDenseVector::multiply(Teuchos::ETransp transa, Teuchos::ETransp transb, double alpha,
      const Teuchos::SerialDenseMatrix<ordinalType, scalarType>& A, const SerialDenseVector& B,
      double beta)
  {
    return vec_.multiply(transa, transb, alpha, A, B.base(), beta);
  }

  void SerialDenseVector::print(std::ostream& out) const { vec_.print(out); }

  SerialDenseVector::Base& SerialDenseVector::base() { return vec_; }
  const SerialDenseVector::Base& SerialDenseVector::base() const { return vec_; }
}  // namespace Core::LinAlg

/*----------------------------------------------------------------------*
 |  Compute vector 2-norm                                               |
 *----------------------------------------------------------------------*/
double Core::LinAlg::norm2(const Core::LinAlg::SerialDenseVector& v) { return v.norm2(); }

/*----------------------------------------------------------------------*
 |  b = alpha*a + beta*b                                                |
 *----------------------------------------------------------------------*/
void Core::LinAlg::update(double alpha, const Core::LinAlg::SerialDenseVector& a, double beta,
    Core::LinAlg::SerialDenseVector& b)
{
  b.scale(beta);
  Core::LinAlg::SerialDenseVector acopy(a);
  acopy.scale(alpha);
  b += acopy;
}

FOUR_C_NAMESPACE_CLOSE
