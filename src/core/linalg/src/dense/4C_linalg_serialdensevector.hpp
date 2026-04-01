// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SERIALDENSEVECTOR_HPP
#define FOUR_C_LINALG_SERIALDENSEVECTOR_HPP


#include "4C_config.hpp"

#include <Teuchos_DataAccess.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseVector.hpp>

#include <ostream>
#include <span>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   * \brief A wrapper around Teuchos::SerialDenseVector
   */
  class SerialDenseVector
  {
   public:
    using ordinalType = int;
    using scalarType = double;
    using Base = Teuchos::SerialDenseVector<ordinalType, scalarType>;

    // --- constructors ---
    SerialDenseVector() = default;
    SerialDenseVector(int length);
    SerialDenseVector(int length, bool zeroOut);
    SerialDenseVector(Teuchos::DataAccess cv, double* values, int length);
    SerialDenseVector(const Base& source, Teuchos::ETransp trans = Teuchos::NO_TRANS);

    // NOLINTBEGIN(readability-identifier-naming)

    // --- size queries ---
    [[nodiscard]] int length() const;
    [[nodiscard]] int num_rows() const;
    [[nodiscard]] int numRows() const;
    [[nodiscard]] double normInf() const;
    [[nodiscard]] double normOne() const;
    [[nodiscard]] double normFrobenius() const;
    [[nodiscard]] bool empty() const;

    // --- element access ---
    double& operator()(int i) { return vec_(i); }
    const double& operator()(int i) const { return vec_(i); }
    double& operator[](int i) { return vec_[i]; }
    const double& operator[](int i) const { return vec_[i]; }

    // --- data access ---
    //! Returns a pointer to the raw data.
    double* values() const;
    //! Returns a pointer to the raw data.
    double* data() const;

    // --- modifiers ---
    int size(int length);
    int Size(int length);
    int resize(int length);
    int scale(double alpha);
    int put_scalar(double val = 0.0);
    int putScalar(double val = 0.0);
    void assign(const SerialDenseVector& source);

    // --- algebraic operations ---
    SerialDenseVector& operator+=(const SerialDenseVector& other);
    double dot(const SerialDenseVector& other) const;
    int multiply(Teuchos::ETransp transa, Teuchos::ETransp transb, double alpha,
        const Teuchos::SerialDenseMatrix<ordinalType, scalarType>& A,
        const Teuchos::SerialDenseMatrix<ordinalType, scalarType>& B, double beta);
    int multiply(Teuchos::ETransp transa, Teuchos::ETransp transb, double alpha,
        const Teuchos::SerialDenseMatrix<ordinalType, scalarType>& A, const SerialDenseVector& B,
        double beta);
    void print(std::ostream& out) const;

    // NOLINTEND(readability-identifier-naming)

    // --- access to underlying Trilinos object ---
    Base& base();
    const Base& base() const;

    /**
     * Get the vector as a std::span.
     */
    [[nodiscard]] std::span<const double> as_span() const { return std::span(values(), length()); }

    /**
     * Get the vector as a std::span.
     */
    [[nodiscard]] std::span<double> as_span() { return std::span(values(), length()); }

   private:
    Base vec_;
  };

  // type definition for serial integer vector
  using IntSerialDenseVector = Teuchos::SerialDenseVector<int, int>;

  /*!
    \brief Update vector components with scaled values of a,
           b = alpha*a + beta*b
    */
  void update(double alpha, const SerialDenseVector& a, double beta, SerialDenseVector& b);

  // wrapper function to compute Norm of vector
  double norm2(const SerialDenseVector& v);

  // output stream operator
  inline std::ostream& operator<<(std::ostream& out, const SerialDenseVector& vec)
  {
    vec.print(out);
    return out;
  }

  // output stream operator
  inline std::ostream& operator<<(std::ostream& out, const IntSerialDenseVector& vec)
  {
    vec.print(out);
    return out;
  }
}  // namespace Core::LinAlg


FOUR_C_NAMESPACE_CLOSE

#endif
