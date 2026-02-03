// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_beaminteraction_contact_input.hpp"

#include "4C_beaminteraction_contact_beam_to_beam_input.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_edge_params.hpp"
#include "4C_beaminteraction_contact_beam_to_sphere_input.hpp"
#include "4C_io_input_spec_builders.hpp"


FOUR_C_NAMESPACE_OPEN

std::vector<Core::IO::InputSpec> BeamInteraction::valid_parameters_contact()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;

  // get parameters for beam to beam contact
  std::vector<Core::IO::InputSpec> beam_to_beam_contact_specs =
      BeamInteraction::Contact::BeamToBeam::valid_parameters();
  specs.insert(specs.end(), beam_to_beam_contact_specs.begin(), beam_to_beam_contact_specs.end());

  // get parameters for beam to solid interaction
  std::vector<Core::IO::InputSpec> beam_to_solid_contact = BeamToSolid::valid_parameters();
  specs.insert(specs.end(), beam_to_solid_contact.begin(), beam_to_solid_contact.end());

  // get parameters for beam to sphere contact
  std::vector<Core::IO::InputSpec> beam_to_sphere_contact_specs =
      BeamInteraction::valid_parameters_contact_beam_to_sphere();
  specs.insert(
      specs.end(), beam_to_sphere_contact_specs.begin(), beam_to_sphere_contact_specs.end());

  return specs;
}

FOUR_C_NAMESPACE_CLOSE