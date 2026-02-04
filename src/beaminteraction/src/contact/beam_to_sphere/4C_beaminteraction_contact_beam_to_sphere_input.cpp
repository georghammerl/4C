// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_beam_to_sphere_input.hpp"

#include "4C_beaminteraction_input.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN



std::vector<Core::IO::InputSpec> BeamInteraction::valid_parameters_contact_beam_to_sphere()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;

  specs.push_back(group("BEAM INTERACTION/BEAM TO SPHERE CONTACT",
      {

          deprecated_selection<BeamInteraction::Strategy>("STRATEGY",
              {
                  {"None", bstr_none},
                  {"none", bstr_none},
                  {"Penalty", bstr_penalty},
                  {"penalty", bstr_penalty},
              },
              {.description = "Type of employed solving strategy", .default_value = bstr_none}),

          parameter<double>("PENALTY_PARAMETER",
              {.description = "Penalty parameter for beam-to-rigidsphere contact",
                  .default_value = 0.0})},
      {.required = false}));

  return specs;
}

FOUR_C_NAMESPACE_CLOSE