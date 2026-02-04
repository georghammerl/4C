// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_spherebeamlinking_input.hpp"

#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN



std::vector<Core::IO::InputSpec> BeamInteraction::valid_parameters_spherebeamlinking()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;

  specs.push_back(group("BEAM INTERACTION/SPHERE BEAM LINK",
      {

          parameter<bool>(
              "SPHEREBEAMLINKING", {.description = "Integrins in problem", .default_value = false}),

          // Reading double parameter for contraction rate for active linker
          parameter<double>("CONTRACTIONRATE",
              {.description = "contraction rate of cell (integrin linker) in [microm/s]",
                  .default_value = 0.0}),
          // time step for stochastic events concerning sphere beam linking
          parameter<double>("TIMESTEP",
              {.description = "time step for stochastic events concerning sphere beam linking "
                              "(e.g. catch-slip-bond behavior) ",
                  .default_value = -1.0}),
          parameter<std::string>("MAXNUMLINKERPERTYPE",
              {.description = "number of crosslinker of certain type ", .default_value = "0"}),
          // material number characterizing crosslinker type
          parameter<std::string>("MATLINKERPERTYPE",
              {.description = "material number characterizing crosslinker type ",
                  .default_value = "-1"}),
          // distance between two binding spots on a filament (same on all filaments)
          parameter<std::string>("FILAMENTBSPOTINTERVALGLOBAL",
              {.description = "distance between two binding spots on all filaments",
                  .default_value = "-1.0"}),
          // distance between two binding spots on a filament (as percentage of current filament
          // length)
          parameter<std::string>("FILAMENTBSPOTINTERVALLOCAL",
              {.description = "distance between two binding spots on current filament",
                  .default_value = "-1.0"}),
          // start and end for bspots on a filament in arc parameter (same on each filament
          // independent of their length)
          parameter<std::string>("FILAMENTBSPOTRANGEGLOBAL",
              {.description = "Lower and upper arc parameter bound for binding spots on a filament",
                  .default_value = "-1.0 -1.0"}),
          // start and end for bspots on a filament in percent of reference filament length
          parameter<std::string>("FILAMENTBSPOTRANGELOCAL",
              {.description = "Lower and upper arc parameter bound for binding spots on a filament",
                  .default_value = "0.0 1.0"})},
      {.required = false}));

  return specs;
}

FOUR_C_NAMESPACE_CLOSE