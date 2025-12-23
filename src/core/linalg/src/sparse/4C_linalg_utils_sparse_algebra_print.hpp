// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_PRINT_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_PRINT_HPP

#include "4C_config.hpp"

#include "4C_linalg_graph.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN

// Forward declarations
namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace Core::LinAlg
{
  //! Print content of @p sparsematrix in Matlab format to file @p filename. Create new file or
  //! overwrite existing one if @p newfile is true
  void print_matrix_in_matlab_format(const std::string& filename,
      const Core::LinAlg::SparseMatrix& sparsematrix, const bool newfile = true);

  //! Print content of @p blockmatrix in Matlab format to file @p filename
  void print_block_matrix_in_matlab_format(
      const std::string& filename, const BlockSparseMatrixBase& blockmatrix);

  //! Print content of @p vector in Matlab format to file @p filename. Create new file or overwrite
  //! existing one if @p newfile is true
  void print_vector_in_matlab_format(const std::string& filename,
      const Core::LinAlg::Vector<double>& vector, const bool newfile = true);

  //! Print content of @p map in Matlab format to file @p filename. Create new file or overwrite
  //! existing one if @p newfile is true
  void print_map_in_matlab_format(
      const std::string& filename, const Core::LinAlg::Map& map, const bool newfile = true);

  /**
   * @brief Read a Matrix Market file and construct a distributed sparse matrix.
   *
   * This function reads a matrix stored in the standard **Matrix Market (.mtx)**
   * format from disk and converts it into a Core::LinAlg::SparseMatrix.
   * The matrix is distributed across MPI processes according to the communicator provided.
   *
   * The function assumes the Matrix Market file represents a sparse matrix and that all MPI ranks
   * in the communicator collectively participate in the read and construction process.
   *
   * @param filename Path to the Matrix Market (.mtx) file to be read.
   * @param comm MPI communicator over which the sparse matrix will be distributed.
   *
   * @return A Core::LinAlg::SparseMatrix containing the data read from the file, distributed
   *         across the given MPI communicator.
   */
  Core::LinAlg::SparseMatrix read_matrix_market_file_as_sparse_matrix(
      const std::string& filename, MPI_Comm comm);


}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
