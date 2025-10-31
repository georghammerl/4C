// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_graph.hpp"

#include "4C_utils_epetra_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Core::LinAlg::Graph::Graph(const Epetra_CrsGraph& Source)
    : graphtype_(CRS_GRAPH), graph_(std::make_unique<Epetra_CrsGraph>(Source))
{
}

Core::LinAlg::Graph::Graph(const Epetra_FECrsGraph& Source)
    : graphtype_(CRS_GRAPH), graph_(std::make_unique<Epetra_CrsGraph>(Source))
{
}

Core::LinAlg::Graph::Graph(const Map& RowMap, const int* NumIndicesPerRow, GraphType graphtype)
    : graphtype_(graphtype)
{
  if (graphtype_ == CRS_GRAPH)
    graph_ =
        std::make_unique<Epetra_CrsGraph>(::Copy, RowMap.get_epetra_block_map(), NumIndicesPerRow);
  else if (graphtype_ == FE_GRAPH)
    graph_ = std::make_unique<Epetra_FECrsGraph>(
        ::Copy, RowMap.get_epetra_block_map(), const_cast<int*>(NumIndicesPerRow));
}

Core::LinAlg::Graph::Graph(const Map& RowMap, int NumIndicesPerRow, GraphType graphtype)
    : graphtype_(graphtype)
{
  if (graphtype_ == CRS_GRAPH)
    graph_ =
        std::make_unique<Epetra_CrsGraph>(::Copy, RowMap.get_epetra_block_map(), NumIndicesPerRow);
  else if (graphtype_ == FE_GRAPH)
    graph_ = std::make_unique<Epetra_FECrsGraph>(
        ::Copy, RowMap.get_epetra_block_map(), NumIndicesPerRow);
}

Core::LinAlg::Graph::Graph(const Graph& other)
    : graphtype_(CRS_GRAPH), graph_(std::make_unique<Epetra_CrsGraph>(other.get_epetra_crs_graph()))
{
}

Core::LinAlg::Graph& Core::LinAlg::Graph::operator=(const Graph& other)
{
  *graph_ = other.get_epetra_crs_graph();
  return *this;
}

void Core::LinAlg::Graph::fill_complete()
{
  if (graphtype_ == CRS_GRAPH)
  {
    CHECK_EPETRA_CALL(graph_->FillComplete());
  }
  else if (graphtype_ == FE_GRAPH)
  {
    CHECK_EPETRA_CALL(static_cast<Epetra_FECrsGraph*>(graph_.get())->GlobalAssemble());
  }
}

void Core::LinAlg::Graph::fill_complete(const Map& domain_map, const Map& range_map)
{
  if (graphtype_ == CRS_GRAPH)
  {
    CHECK_EPETRA_CALL(
        graph_->FillComplete(domain_map.get_epetra_block_map(), range_map.get_epetra_block_map()));
  }
  else if (graphtype_ == FE_GRAPH)
  {
    CHECK_EPETRA_CALL(static_cast<Epetra_FECrsGraph*>(graph_.get())
            ->GlobalAssemble(domain_map.get_epetra_map(), range_map.get_epetra_map()));
  }
}

void Core::LinAlg::Graph::optimize_storage() { CHECK_EPETRA_CALL(graph_->OptimizeStorage()); }

void Core::LinAlg::Graph::export_to(const Core::LinAlg::Graph& A,
    const Core::LinAlg::Export& Exporter, Epetra_CombineMode CombineMode)
{
  CHECK_EPETRA_CALL(
      graph_->Export(A.get_epetra_crs_graph(), Exporter.get_epetra_export(), CombineMode));
}

void Core::LinAlg::Graph::import_from(const Core::LinAlg::Graph& A,
    const Core::LinAlg::Import& Importer, Epetra_CombineMode CombineMode)
{
  CHECK_EPETRA_CALL(
      graph_->Import(A.get_epetra_crs_graph(), Importer.get_epetra_import(), CombineMode));
}

void Core::LinAlg::Graph::insert_global_indices(int GlobalRow, int NumIndices, int* Indices)
{
  CHECK_EPETRA_CALL(graph_->InsertGlobalIndices(GlobalRow, NumIndices, Indices));
}

void Core::LinAlg::Graph::insert_global_indices(
    int numRows, const int* rows, int numCols, const int* cols)
{
  if (graphtype_ == CRS_GRAPH)
  {
    FOUR_C_THROW("This type of insert_global_indices() is only available for FE_GRAPH type.");
  }
  else if (graphtype_ == FE_GRAPH)
  {
    CHECK_EPETRA_CALL(static_cast<Epetra_FECrsGraph*>(graph_.get())
            ->InsertGlobalIndices(numRows, rows, numCols, cols));
  }
}

void Core::LinAlg::Graph::extract_local_row_view(int LocalRow, std::span<int>& Indices) const
{
  int num_indices;
  int* indices;
  CHECK_EPETRA_CALL(graph_->ExtractMyRowView(LocalRow, num_indices, indices));
  Indices = std::span(indices, num_indices);
}

void Core::LinAlg::Graph::extract_global_row_view(int GlobalRow, std::span<int>& Indices) const
{
  int num_indices;
  int* indices;
  CHECK_EPETRA_CALL(graph_->ExtractGlobalRowView(GlobalRow, num_indices, indices));
  Indices = std::span(indices, num_indices);
}

void Core::LinAlg::Graph::extract_global_row_copy(
    int GlobalRow, int LenOfIndices, int& NumIndices, int* Indices) const
{
  CHECK_EPETRA_CALL(graph_->ExtractGlobalRowCopy(GlobalRow, LenOfIndices, NumIndices, Indices));
}

FOUR_C_NAMESPACE_CLOSE
