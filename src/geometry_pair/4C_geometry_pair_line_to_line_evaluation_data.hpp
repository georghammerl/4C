// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_LINE_TO_LINE_EVALUATION_DATA_HPP
#define FOUR_C_GEOMETRY_PAIR_LINE_TO_LINE_EVALUATION_DATA_HPP


#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_geometry_pair_evaluation_data_base.hpp"
#include "4C_geometry_pair_input.hpp"
#include "4C_geometry_pair_utility_classes.hpp"


FOUR_C_NAMESPACE_OPEN

namespace GeometryPair
{
  /**
   * \brief Class to manage inout parameters and evaluation data for line to line interactions.
   */
  class LineToLineEvaluationData : public GeometryEvaluationDataBase
  {
   public:
    /**
     * \brief Constructor (derived).
     */
    LineToLineEvaluationData(const Core::FE::Discretization& discretization,
        const Core::Conditions::Condition* condition_1,
        const Core::Conditions::Condition* condition_2);

    /**
     * \brief Clear the evaluation data.
     */
    void clear() override;

    /**
     * \brief Check if a given set of projection coordinates for a given element pair should be
     * evaluated.
     *
     * This function allows to avoid double evaluations of CPPs that lie directly on nodes.
     */
    [[nodiscard]] bool evaluate_projection_coordinates(
        const std::array<const Core::Elements::Element*, 2>& pair_elements,
        const std::array<double, 2>& position_in_parameterspace) const;

   private:
    /// Map from conditions node IDs to minimum connected element GIDs. This is used to ensure
    /// that CPP directly on nodes are not evaluated twice.
    std::array<std::map<int, int>, 2> condition_node_to_min_element_gid_map_{};
  };
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
