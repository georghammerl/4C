// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization_builder.hpp"

#include "4C_rebalance.hpp"

FOUR_C_NAMESPACE_OPEN

template <int dim>
void Core::FE::DiscretizationBuilder<dim>::add_node(std::span<const double, dim> x,
    IndexType global_id, std::shared_ptr<Core::Nodes::Node> user_node)
{
  FOUR_C_ASSERT_ALWAYS(!nodes_.contains(global_id),
      "Node with global id {} already added to the builder.", global_id);

  std::array<double, dim> coordinates;
  std::copy(x.begin(), x.end(), coordinates.begin());
  nodes_.emplace(global_id, NodeData{
                                .x = coordinates,
                                .global_id = global_id,
                                .user_node = user_node,
                            });
  used_node_ids_.insert(global_id);
}



template <int dim>
void Core::FE::DiscretizationBuilder<dim>::add_element(Core::FE::CellType cell_type,
    std::span<const IndexType> node_ids, IndexType global_id,
    std::shared_ptr<Core::Elements::Element> user_element)
{
  FOUR_C_ASSERT_ALWAYS(!elements_.contains(global_id),
      "Element with global id {} already added to the builder.", global_id);

  FOUR_C_ASSERT_ALWAYS(Core::FE::num_nodes(cell_type) == static_cast<int>(node_ids.size()),
      "Tried to add element with global id {} with {} nodes, but cell type {} expects {} "
      "nodes.",
      global_id, node_ids.size(), Core::FE::cell_type_to_string(cell_type),
      Core::FE::num_nodes(cell_type));

  std::vector<IndexType> node_ids_copy(node_ids.size());
  std::ranges::copy(node_ids, node_ids_copy.begin());
  elements_.emplace(global_id, ElementData{
                                   .cell_type = cell_type,
                                   .node_ids = std::move(node_ids_copy),
                                   .global_id = global_id,
                                   .user_element = user_element,
                               });
}



template <int dim>
void Core::FE::DiscretizationBuilder<dim>::build(Discretization& discretization,
    const Core::Rebalance::RebalanceParameters& rebalance_parameters)
{
  assert_consistent();

  IO::MeshInput::RawMesh<dim> mesh;
  std::map<int, std::shared_ptr<Core::Nodes::Node>> user_nodes;
  std::map<int, std::shared_ptr<Core::Elements::Element>> user_elements;

  const int my_rank = Core::Communication::my_mpi_rank(discretization.get_comm());
  if (my_rank == 0)
  {
    mesh.points.reserve(nodes_.size());
    mesh.external_ids.emplace();

    for (const auto& [nid, node] : nodes_)
    {
      mesh.points.push_back(node.x);
      mesh.external_ids->push_back(node.global_id);
      if (node.user_node) user_nodes.emplace(nid, node.user_node);
    }

    std::unordered_map<CellType, IO::MeshInput::CellBlock<dim>> cell_blocks;

    const auto ensure_cell_block_exists = [&cell_blocks](
                                              CellType cell_type) -> IO::MeshInput::CellBlock<dim>&
    {
      if (!cell_blocks.contains(cell_type))
      {
        cell_blocks.emplace(cell_type, IO::MeshInput::CellBlock<dim>(cell_type));
        cell_blocks.at(cell_type).external_ids_.emplace();
        cell_blocks.at(cell_type).name.emplace(
            "auto_clustered_block_for_" + cell_type_to_string(cell_type));
      }
      return cell_blocks.at(cell_type);
    };

    for (const auto& [eid, elem] : elements_)
    {
      auto& cell_block = ensure_cell_block_exists(elem.cell_type);

      cell_block.add_cell(elem.node_ids);
      cell_block.external_ids_->push_back(elem.global_id);

      if (elem.user_element) user_elements.emplace(eid, elem.user_element);
    }

    int cell_block_index = 0;
    for (auto&& [cell_type, cell_block] : cell_blocks)
    {
      mesh.cell_blocks.emplace(cell_block_index, std::move(cell_block));
      cell_block_index++;
    }
    assert_valid(mesh);
  }

  IO::MeshInput::Mesh<dim> final_mesh(std::move(mesh));
  discretization.fill_from_mesh(final_mesh, user_elements, user_nodes, rebalance_parameters);
}



template <int dim>
void Core::FE::DiscretizationBuilder<dim>::assert_consistent() const
{
  // Check that all element node ids refer to nodes that have been added to the builder
  // Record all nodes that are referenced by at least one element
  std::unordered_set<IndexType> node_ids_referenced_by_elements;
  for (const auto& [eid, elem] : elements_)
  {
    for (const auto nid : elem.node_ids)
    {
      node_ids_referenced_by_elements.insert(nid);
      if (!used_node_ids_.contains(nid))
      {
        FOUR_C_THROW(
            "Element with global id {} refers to node id {} that was not added to the builder.",
            eid, nid);
      }
    }
  }

  // Check that all added nodes have indeed been referenced by at least one element
  for (const auto nid : used_node_ids_)
  {
    if (!node_ids_referenced_by_elements.contains(nid))
    {
      FOUR_C_THROW(
          "Node with global id {} was added to the builder, but is not used by any element.", nid);
    }
  }
}

template class Core::FE::DiscretizationBuilder<3>;

FOUR_C_NAMESPACE_CLOSE
