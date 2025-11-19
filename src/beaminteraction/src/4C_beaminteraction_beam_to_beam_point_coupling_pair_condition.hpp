// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_POINT_COUPLING_PAIR_CONDITION_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_POINT_COUPLING_PAIR_CONDITION_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_beam_point_coupling_pair.hpp"
#include "4C_beaminteraction_conditions.hpp"
#include "4C_geometry_pair_line_to_line_evaluation_data.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief This base class represents a single beam point coupling condition with direct creation.
   */
  class BeamToBeamPointCouplingConditionDirect : public BeamInteractionConditionBase
  {
   public:
    BeamToBeamPointCouplingConditionDirect(const Core::Conditions::Condition& condition_line,
        double positional_penalty_parameter, double rotational_penalty_parameter)
        : BeamInteractionConditionBase(condition_line),
          positional_penalty_parameter_(positional_penalty_parameter),
          rotational_penalty_parameter_(rotational_penalty_parameter)
    {
    }

    /**
     * \brief Check if a combination of beam ids is in this condition.
     */
    bool ids_in_condition(const int id_line, const int id_other) const override { return false; };

    /**
     * \brief Create the beam contact pairs needed for this condition (derived).
     */
    std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) override
    {
      return nullptr;
    }

    /**
     * \brief Create the contact pairs directly, i.e., not depending on given element pointers
     * (derived).
     */
    void create_contact_pairs_direct(std::vector<std::shared_ptr<BeamContactPair>>& contact_pairs,
        const Core::FE::Discretization& discretization,
        const std::shared_ptr<BeamInteraction::BeamContactParams>& params_ptr) override;

   private:
    /// Penalty parameter used to couple the positional DoFs
    double positional_penalty_parameter_;
    /// Penalty parameter used to couple the rotational DoFs
    double rotational_penalty_parameter_;
  };

  /**
   * \brief This base class represents a single beam point coupling condition with direct creation.
   */
  class BeamToBeamPointCouplingConditionIndirect : public BeamInteractionConditionBase
  {
   public:
    /**
     * \brief Constructor.
     *
     * @param condition_line_1 (in) The first line condition containing the beam elements.
     * @param condition_line_2 (in) The other line condition containing the beam elements.
     */
    BeamToBeamPointCouplingConditionIndirect(const Core::Conditions::Condition& condition_line_1,
        const Core::Conditions::Condition& condition_line_2)
        : BeamInteractionConditionBase(condition_line_1),
          condition_other_(&condition_line_2),
          other_line_ids_()
    {
      auto init_parameters = [this](const Core::Conditions::Condition& condition) -> bool
      {
        if (condition.parameters().has_group("PARAMETERS"))
        {
          const auto parameters = condition.parameters().group("PARAMETERS");
          parameters_.penalty_parameter_pos =
              parameters.get<double>("POSITIONAL_PENALTY_PARAMETER");
          parameters_.penalty_parameter_rot =
              parameters.get<double>("ROTATIONAL_PENALTY_PARAMETER");
          parameters_.projection_valid_factor = parameters.get<double>("PROJECTION_VALID_FACTOR");
          parameters_.use_closest_point_projection = true;
          return true;
        }
        return false;
      };

      const bool parameters_in_condition_1 = init_parameters(condition_line_1);
      const bool parameters_in_condition_2 = init_parameters(condition_line_2);

      if (parameters_in_condition_1 && parameters_in_condition_2)
      {
        FOUR_C_THROW(
            "Parameters for BeamToBeamPointCouplingConditionIndirect defined in both conditions.");
      }
      else if (!parameters_in_condition_1 && !parameters_in_condition_2)
      {
        FOUR_C_THROW(
            "Parameters for BeamToBeamPointCouplingConditionIndirect not defined in either "
            "condition.");
      }
    };

    /**
     * \brief Build the ID sets for this condition.
     *
     * The BuildIdSets method from the base class is called to build the beam IDs.
     */
    void build_id_sets(
        const std::shared_ptr<const Core::FE::Discretization>& discretization) override;

    /**
     * \brief Check if a combination of beam and beam id is in this condition.
     */
    bool ids_in_condition(const int id_line, const int id_other) const override;

    /**
     * \brief Create the beam to beam pairs needed for this condition (derived).
     */
    std::shared_ptr<BeamInteraction::BeamContactPair> create_contact_pair(
        const std::vector<Core::Elements::Element const*>& ele_ptrs) override;

   protected:
    /**
     * \brief Check if a ID is in a condition.
     */
    inline static bool id_is_in_condition(const std::set<int>& id_set, const int id)
    {
      return id_set.find(id) != id_set.end();
    }

   private:
    //! Pointer to the other line condition.
    const Core::Conditions::Condition* condition_other_;

    //! Set containing the other line element IDs.
    std::set<int> other_line_ids_;

    // Parameters for the created coupling pairs.
    BeamToBeamPointCouplingPairParameters parameters_{};

    /// Pointer to geometry evaluation data for line-to-line projections
    std::shared_ptr<GeometryPair::LineToLineEvaluationData> geometry_evaluation_data_;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
