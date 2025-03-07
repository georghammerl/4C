// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GLOBAL_LEGACY_MODULE_VALIDMATERIALS_HPP
#define FOUR_C_GLOBAL_LEGACY_MODULE_VALIDMATERIALS_HPP

#include "4C_config.hpp"

#include <Teuchos_Array.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class MaterialDefinition;
}

namespace Global
{
  /// construct list with all materials and documentation
  std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> valid_materials();
}  // namespace Global


FOUR_C_NAMESPACE_CLOSE

#endif
