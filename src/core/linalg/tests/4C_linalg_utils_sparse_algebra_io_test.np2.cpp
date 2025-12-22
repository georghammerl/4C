// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_sparse_algebra_io.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace
{

  TEST(SparseAlgebraIO, ReadMatrixMarketFileAsMultiVector)
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = Core::Communication::my_mpi_rank(comm);

    Core::LinAlg::Map map{10, 5, 0, comm};
    Core::LinAlg::MultiVector<double> mv = Core::LinAlg::read_matrix_market_file_as_multi_vector(
        TESTING::get_support_file_path("test_matrices/matrix_market.mm"), map);

    EXPECT_EQ(mv.local_length(), 5);
    EXPECT_EQ(mv.num_vectors(), 2);
    if (rank == 0)
    {
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[0], 0.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[1], 1.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[2], 2.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[3], 3.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[4], 4.0, 1e-18);

      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[0], 0.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[1], 1.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[2], 2.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[3], 3.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[4], 4.1, 1e-18);
    }
    else if (rank == 1)
    {
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[0], 5.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[1], 6.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[2], 7.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[3], 8.0, 1e-18);
      EXPECT_NEAR(mv.get_vector(0).local_values_as_span()[4], 9.0, 1e-18);

      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[0], 5.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[1], 6.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[2], 7.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[3], 8.1, 1e-18);
      EXPECT_NEAR(mv.get_vector(1).local_values_as_span()[4], 9.1, 1e-18);
    }
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE
