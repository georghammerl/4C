// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_model_registry.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit_elasticity_registry.hpp"
#include "4C_reduced_lung_terminal_unit_rheology_registry.hpp"
#include "4C_utils_exceptions.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    using namespace TerminalUnits::ModelRegistry;

    const std::vector<TerminalUnitModelKey>& compatible_terminal_unit_model_pairs()
    {
      static const std::vector<TerminalUnitModelKey> compatible_pairs = {
          {RheologicalModelType::KelvinVoigt, ElasticityModelType::Linear},
          {RheologicalModelType::KelvinVoigt, ElasticityModelType::Ogden},
          {RheologicalModelType::FourElementMaxwell, ElasticityModelType::Linear},
          {RheologicalModelType::FourElementMaxwell, ElasticityModelType::Ogden},
      };

      return compatible_pairs;
    }

    const TerminalUnitFactoryMap& terminal_unit_factory_registry()
    {
      static const TerminalUnitFactoryMap registry = []
      {
        TerminalUnitFactoryMap factories;

        for (const auto& [rheological_model_type, elasticity_model_type] :
            compatible_terminal_unit_model_pairs())
        {
          const auto [it, inserted] = factories.emplace(
              TerminalUnitModelKey{rheological_model_type, elasticity_model_type},
              [rheological_model_type, elasticity_model_type](
                  TerminalUnits::TerminalUnitContainer& terminal_units, int global_element_id,
                  int local_element_id, const ReducedLungParameters& parameters)
              {
                dispatch_rheological_model_type(rheological_model_type,
                    [&]<typename RheologicalModel>()
                    {
                      dispatch_elasticity_model_type(elasticity_model_type,
                          [&]<typename ElasticityModel>()
                          {
                            TerminalUnits::add_terminal_unit_ele<RheologicalModel, ElasticityModel>(
                                terminal_units, global_element_id, local_element_id, parameters);
                          });
                    });
              });

          if (!inserted)
          {
            FOUR_C_THROW(
                "Duplicate terminal-unit model registration for (rheological='{}', "
                "elasticity='{}').",
                rheological_model_name(rheological_model_type),
                elasticity_model_name(elasticity_model_type));
          }
          FOUR_C_ASSERT_ALWAYS(it != factories.end(), "Invalid terminal-unit registry insertion.");
        }

        return factories;
      }();

      return registry;
    }
  }  // namespace

  void add_terminal_unit_with_model_selection(TerminalUnits::TerminalUnitContainer& terminal_units,
      int global_element_id, int local_element_id, const ReducedLungParameters& parameters,
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType
          rheological_model_type,
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType
          elasticity_model_type)
  {
    const TerminalUnitModelKey key{rheological_model_type, elasticity_model_type};
    const auto& registry = terminal_unit_factory_registry();
    const auto factory_it = registry.find(key);
    if (factory_it == registry.end())
    {
      FOUR_C_THROW(
          "Terminal unit model combination not implemented (rheological='{}', elasticity='{}').",
          rheological_model_name(rheological_model_type),
          elasticity_model_name(elasticity_model_type));
    }

    factory_it->second(terminal_units, global_element_id, local_element_id, parameters);
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
