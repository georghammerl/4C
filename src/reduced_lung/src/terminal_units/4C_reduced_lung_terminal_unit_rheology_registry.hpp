// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_RHEOLOGY_REGISTRY_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_RHEOLOGY_REGISTRY_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_model_registry.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::ModelRegistry
{
  inline const char* rheological_model_name(const RheologicalModelType rheological_model_type)
  {
    switch (rheological_model_type)
    {
      case RheologicalModelType::KelvinVoigt:
        return "KelvinVoigt";
      case RheologicalModelType::FourElementMaxwell:
        return "FourElementMaxwell";
    }
    FOUR_C_THROW("Unknown rheological model type enum value.");
  }

  template <typename Callable>
  void dispatch_rheological_model_type(
      const RheologicalModelType rheological_model_type, Callable&& callable)
  {
    switch (rheological_model_type)
    {
      case RheologicalModelType::KelvinVoigt:
        std::forward<Callable>(callable).template operator()<KelvinVoigt>();
        return;
      case RheologicalModelType::FourElementMaxwell:
        std::forward<Callable>(callable).template operator()<FourElementMaxwell>();
        return;
    }
    FOUR_C_THROW("Unknown rheological model type enum value.");
  }
}  // namespace ReducedLung::TerminalUnits::ModelRegistry

FOUR_C_NAMESPACE_CLOSE

#endif
