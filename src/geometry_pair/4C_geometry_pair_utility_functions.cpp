// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_utility_functions.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
std::string GeometryPair::discretization_type_geometry_to_string(
    const DiscretizationTypeGeometry discretization_type)
{
  switch (discretization_type)
  {
    case DiscretizationTypeGeometry::none:
      return "undefined";
    case DiscretizationTypeGeometry::triangle:
      return "triangle";
    case DiscretizationTypeGeometry::quad:
      return "quadrilateral";
    case DiscretizationTypeGeometry::hexahedron:
      return "hexahedron";
    case DiscretizationTypeGeometry::tetraeder:
      return "tetraeder";
    default:
      FOUR_C_THROW(
          "GeometryPair::DiscretizationTypeGeometryToString: Got unexpected discretization "
          "type.");
      return "";
      break;
  }
}

/**
 *
 */
std::unordered_map<int, const Core::Elements::Element*> GeometryPair::condition_to_element_id_map(
    const Core::Conditions::Condition& condition)
{
  std::unordered_map<int, const Core::Elements::Element*> element_id_map;

  for (const auto& [_, element] : condition.geometry())
  {
    if (element->is_face_element())
    {
      // For a face element it is easy as we can directly get the parent element ID.
      const std::shared_ptr<const Core::Elements::FaceElement> face_element =
          std::dynamic_pointer_cast<const Core::Elements::FaceElement>(element);
      const int parent_element_id = face_element->parent_element_id();
      element_id_map[parent_element_id] = face_element.get();
    }
    else
    {
      const size_t n_nodes = element->num_node();

      // Create the node sets and store the node IDs from the condition element in it.
      std::set<int> nodes_condition;
      std::set<int> nodes_element;
      for (size_t i = 0; i < n_nodes; i++) nodes_condition.insert(element->nodes()[i]->id());

      // Loop over all elements connected to a node and check if the nodal IDs are the same.
      const size_t local_node_id = n_nodes - 1;
      for (auto ele : element->nodes()[local_node_id]->adjacent_elements())
      {
        if (ele.num_nodes() != n_nodes) continue;

        // Fill up the node ID map.
        nodes_element.clear();
        for (auto node : ele.nodes()) nodes_element.insert(node.global_id());

        // Check if the maps are equal.
        if (std::equal(nodes_condition.begin(), nodes_condition.end(), nodes_element.begin()))
        {
          element_id_map[ele.global_id()] = ele.user_element();
          break;
        }
      }
    }
  }

  if (element_id_map.size() != condition.geometry().size())
    FOUR_C_THROW("Could not create all element ID to element pointer mappings for the condition!");

  return element_id_map;
}


FOUR_C_NAMESPACE_CLOSE
