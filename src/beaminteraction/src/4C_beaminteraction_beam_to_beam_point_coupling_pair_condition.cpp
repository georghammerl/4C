// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_beam_point_coupling_pair_condition.hpp"

#include "4C_beaminteraction_beam_to_beam_point_coupling_pair.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometry_pair_element.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
void BeamInteraction::BeamToBeamPointCouplingCondition::clear() {}

/**
 *
 */
void BeamInteraction::BeamToBeamPointCouplingCondition::create_contact_pairs_direct(
    std::vector<std::shared_ptr<BeamContactPair>>& contact_pairs,
    const Core::FE::Discretization& discretization,
    const std::shared_ptr<BeamInteraction::BeamContactParams>& params_ptr)
{
  // Set the IDs of the nodes to be coupled
  const std::vector<int> node_ids = *(condition_line_->get_nodes());

  if (node_ids.size() != 2)
    FOUR_C_THROW(
        "The Penalty Point Coupling Condition can only handle 2 nodes per condition! If you want "
        "to couple multiple nodes, split them into multiple conditions, each coupling two of the "
        "beam nodes.");

  // Get the element with the lowest GID among the adjacent elements of a node
  auto get_lowest_gid_element = [](const auto* node) -> const Core::Elements::Element*
  {
    const auto& adj_elements = node->adjacent_elements();

    const Core::Elements::Element* lowest = nullptr;
    int lowest_gid = std::numeric_limits<int>::max();

    for (const auto& adj_elem : adj_elements)
    {
      const Core::Elements::Element* element = adj_elem.user_element();
      if (element && element->id() < lowest_gid)
      {
        lowest_gid = element->id();
        lowest = element;
      }
    }
    return lowest;
  };

  // We create the pair on the processor that owns the beam element with the lowest GID connected to
  // the first node.
  std::array<const Core::Elements::Element*, 2> element_ptrs{};
  for (size_t i_node = 0; i_node < 2; i_node++)
  {
    if (discretization.have_global_node(node_ids[i_node]))
    {
      element_ptrs[i_node] = get_lowest_gid_element(discretization.g_node(node_ids[i_node]));
    }
  }

  // We check if the first element pointer is valid and if that element is owned by this processor.
  int pairs_created = 0;
  if (element_ptrs[0] != nullptr)
  {
    const auto my_rank = Core::Communication::my_mpi_rank(discretization.get_comm());
    if (element_ptrs[0]->owner() == my_rank)
    {
      // Check that the second element also exists on this processor (can be a ghosted element)
      if (element_ptrs[1] != nullptr)
      {
        // Get the parameter coordinates for evaluating the coupling constraint
        for (size_t i_node = 0; i_node < 2; i_node++)
        {
          const Core::Nodes::Node* node = discretization.g_node(node_ids[i_node]);
          if (element_ptrs[i_node]->node_ids()[0] == node->id())
            local_parameter_coordinates_[i_node] = -1;
          else
            local_parameter_coordinates_[i_node] = 1;
        }

        // Create the pair
        contact_pairs.emplace_back(
            std::make_shared<BeamToBeamPointCouplingPair<GeometryPair::t_hermite>>(
                rotational_penalty_parameter_, positional_penalty_parameter_,
                local_parameter_coordinates_));
        contact_pairs.back()->init(params_ptr, {element_ptrs[0], element_ptrs[1]});
        contact_pairs.back()->setup();
        pairs_created += 1;
      }
      else
      {
        FOUR_C_THROW(
            "The element for node {} is owned by rank {}, but the element for the node {} could "
            "not be found on this rank.",
            node_ids[0], my_rank, node_ids[1]);
      }
    }
  }

  const auto total_created_pairs =
      Core::Communication::sum_all(pairs_created, discretization.get_comm());
  if (total_created_pairs != 1)
  {
    FOUR_C_THROW(
        "BeamToBeamPointCouplingCondition: Expected exactly one contact pair to be created across "
        "all MPI ranks, but found {}.",
        total_created_pairs);
  }
}


FOUR_C_NAMESPACE_CLOSE
