// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CROSSLINKING_INPUT_HPP
#define FOUR_C_BEAMINTERACTION_CROSSLINKING_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{

  /// get crosslinking parameters
  std::vector<Core::IO::InputSpec> valid_parameters_crosslinking();

}  // namespace BeamInteraction


FOUR_C_NAMESPACE_CLOSE

#endif
