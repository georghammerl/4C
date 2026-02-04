// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SPHERE_INPUT_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SPHERE_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{

  /// type of employed solving strategy for contact
  /// (this enum represents the input file parameter STRATEGY)
  enum Strategy
  {
    bstr_none,    ///< no beam contact
    bstr_penalty  ///< penalty method
  };

  /// get beam to sphere contact parameters
  std::vector<Core::IO::InputSpec> valid_parameters_contact_beam_to_sphere();

}  // namespace BeamInteraction


FOUR_C_NAMESPACE_CLOSE

#endif
