// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_MODEL_REGISTRY_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_MODEL_REGISTRY_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit.hpp"

#include <functional>
#include <map>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::ModelRegistry
{
  using RheologicalModelType =
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;
  using ElasticityModelType =
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;

  using TerminalUnitFactory = std::function<void(TerminalUnitContainer& terminal_units,
      int global_element_id, int local_element_id, const ReducedLungParameters& parameters)>;

  using TerminalUnitModelKey = std::pair<RheologicalModelType, ElasticityModelType>;
  using TerminalUnitFactoryMap = std::map<TerminalUnitModelKey, TerminalUnitFactory>;
}  // namespace ReducedLung::TerminalUnits::ModelRegistry

FOUR_C_NAMESPACE_CLOSE

#endif
