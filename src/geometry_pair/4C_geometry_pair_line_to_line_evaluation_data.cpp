// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_line_to_line_evaluation_data.hpp"

#include "4C_geometry_pair_utility_functions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <cstddef>

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
GeometryPair::LineToLineEvaluationData::LineToLineEvaluationData(
    const Core::FE::Discretization& discretization, const Core::Conditions::Condition* condition_1,
    const Core::Conditions::Condition* condition_2)
    : GeometryEvaluationDataBase(), condition_node_to_min_element_gid_map_{}
{
  // Initialize evaluation data structures.
  clear();

  std::array<const Core::Conditions::Condition*, 2> conditions{condition_1, condition_2};

  for (size_t i_condition = 0; i_condition < conditions.size(); ++i_condition)
  {
    const auto element_id_map = condition_to_element_id_map(*conditions[i_condition]);
    auto& node_to_min_ele_map = condition_node_to_min_element_gid_map_.at(i_condition);

    for (const auto& [_, element] : element_id_map)
    {
      // We can use this GID here, as we will also have access to these IDs inside the pair.
      const int element_gid = element->id();
      for (const auto node : element->node_range())
      {
        auto [it, inserted] = node_to_min_ele_map.try_emplace(node.global_id(), element_gid);
        if (!inserted && element_gid < it->second) it->second = element_gid;
      }
    }
  }
}

/**
 *
 */
void GeometryPair::LineToLineEvaluationData::clear()
{
  // Call reset on the base method.
  GeometryEvaluationDataBase::clear();

  // Initialize an empty map for tracking evaluated nodes.
  for (size_t i = 0; i < 2; ++i) condition_node_to_min_element_gid_map_[i].clear();
}

/**
 *
 */
bool GeometryPair::LineToLineEvaluationData::evaluate_projection_coordinates(
    const std::array<const Core::Elements::Element*, 2>& pair_elements,
    const std::array<double, 2>& position_in_parameterspace) const
{
  // Make sure that we have unique pairs for projections on nodes
  for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
  {
    for (unsigned int i_node = 0; i_node < 2; i_node++)
    {
      const double xi = -1.0 + i_node * 2.0;
      if (std::abs(position_in_parameterspace[i_beam] - xi) <
          GeometryPair::Constants::projection_xi_eta_tol)
      {
        // We have a projection directly on a node, check if this node should be evaluated on
        // this pair.
        const auto& node_to_min_ele_map = condition_node_to_min_element_gid_map_[i_beam];
        const int node_id = pair_elements[i_beam]->nodes()[i_node]->id();
        const int element_gid = pair_elements[i_beam]->id();
        const auto map_iter = node_to_min_ele_map.find(node_id);
        if (map_iter != node_to_min_ele_map.end())
        {
          const int min_element_gid = map_iter->second;
          if (element_gid != min_element_gid)
          {
            // This element is not the minimum GID element for this node, skip evaluation.
            return false;
          }
        }
        else
        {
          FOUR_C_THROW(
              "Could not find node ID {} in condition for beam-to-beam point coupling pair "
              "between elements {} and {}.",
              node_id, pair_elements[0]->id(), pair_elements[1]->id());
        }
      }
    }
  }
  return true;
}

FOUR_C_NAMESPACE_CLOSE
