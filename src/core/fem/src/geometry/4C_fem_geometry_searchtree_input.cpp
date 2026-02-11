// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_searchtree_input.hpp"

#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN

Core::IO::InputSpec Geo::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  Core::IO::InputSpec spec = group("SEARCH TREE",
      {

          deprecated_selection<Geo::TreeType>("TREE_TYPE",
              {
                  {"notree", Geo::Notree},
                  {"octree3d", Geo::Octree3D},
                  {"quadtree3d", Geo::Quadtree3D},
                  {"quadtree2d", Geo::Quadtree2D},
              },
              {.description = "set tree type", .default_value = Geo::Notree})},
      {.required = false});
  return spec;
}

FOUR_C_NAMESPACE_CLOSE