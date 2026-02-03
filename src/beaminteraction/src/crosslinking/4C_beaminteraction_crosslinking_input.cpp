// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_crosslinking_input.hpp"

#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN



std::vector<Core::IO::InputSpec> BeamInteraction::valid_parameters_crosslinking()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;

  specs.push_back(group("BEAM INTERACTION/CROSSLINKING",
      {

          // remove this some day
          parameter<bool>(
              "CROSSLINKER", {.description = "Crosslinker in problem", .default_value = false}),

          // bounding box for initial random crosslinker position
          parameter<std::string>("INIT_LINKER_BOUNDINGBOX",
              {.description = "Linker are initially set randomly within this bounding box",
                  .default_value = "1e12 1e12 1e12 1e12 1e12 1e12"}),

          // time step for stochastic events concerning crosslinking
          parameter<double>("TIMESTEP",
              {.description = "time step for stochastic events concerning crosslinking (e.g. "
                              "diffusion, p_link, p_unlink) ",
                  .default_value = -1.0}),
          // Reading double parameter for viscosity of background fluid
          parameter<double>("VISCOSITY", {.description = "viscosity", .default_value = 0.0}),
          // Reading double parameter for thermal energy in background fluid (temperature *
          // Boltzmann
          // constant)
          parameter<double>("KT", {.description = "thermal energy", .default_value = 0.0}),
          // number of initial (are set right in the beginning) crosslinker of certain type
          parameter<std::string>("MAXNUMINITCROSSLINKERPERTYPE",
              {.description = "number of initial crosslinker of certain "
                              "type (additional to NUMCROSSLINKERPERTYPE) ",
                  .default_value = "0"}),
          // number of crosslinker of certain type
          parameter<std::string>("NUMCROSSLINKERPERTYPE",
              {.description = "number of crosslinker of certain type ", .default_value = "0"}),
          // material number characterizing crosslinker type
          parameter<std::string>("MATCROSSLINKERPERTYPE",
              {.description = "material number characterizing crosslinker type ",
                  .default_value = "-1"}),
          // maximal number of binding partner per filament binding spot for each binding spot type
          parameter<std::string>("MAXNUMBONDSPERFILAMENTBSPOT",
              {.description = "maximal number of bonds per filament binding spot",
                  .default_value = "1"}),
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
          // independent
          // of
          // their length)
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