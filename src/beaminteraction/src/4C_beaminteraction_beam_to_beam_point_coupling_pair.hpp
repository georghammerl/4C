// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_POINT_COUPLING_PAIR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_POINT_COUPLING_PAIR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_geometry_pair_line_to_line_evaluation_data.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN


// Forward declarations.
namespace Core::LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}  // namespace Core::LargeRotations


namespace BeamInteraction
{
  /**
   * \brief Parameters for point-wise beam-to-beam mesh tying.
   */
  struct BeamToBeamPointCouplingPairParameters
  {
    //! Penalty parameter for positional coupling.
    double penalty_parameter_pos = 0.0;

    //! Penalty parameter for rotational coupling.
    double penalty_parameter_rot = 0.0;

    //! If the interacting points are computed via closest point projection or not.
    bool use_closest_point_projection = false;

    //! Flag if this pair should be evaluated.
    bool evaluate_pair = true;

    //! Coupling point positions in the element parameter spaces.
    std::array<double, 2> position_in_parameterspace = {0.0, 0.0};

    //! Factor to determine valid projection
    double projection_valid_factor = -1.0;

    //! Flag on type of constraint enforcement.
    enum class ConstraintEnforcement
    {
      penalty_direct,
      penalty_indirect
    };
    ConstraintEnforcement constraint_enforcement = ConstraintEnforcement::penalty_direct;

    //! Maximum number of pairs per beam element for indirect variant.
    unsigned int n_pairs_per_element = 0;
  };

  /**
   * \brief Data container for beam-to-beam kinematics.
   */
  template <unsigned int n_dof_beam_1, unsigned int n_dof_beam_2>
  struct BeamToBeamKinematic
  {
    //! Element displacement vectors.
    std::array<std::vector<double>, 2> element_displacement;

    //! Global DOFs of the coupling pair.
    std::array<int, n_dof_beam_1 + n_dof_beam_2> pair_gid{-1};

    //! Reference and current position of both coupled points.
    std::array<Core::LinAlg::Matrix<3, 1>, 2> r_ref;
    std::array<Core::LinAlg::Matrix<3, 1>, 2> r;

    //! Cross-section orientations (reference and current) for both coupled points.
    std::array<Core::LinAlg::Matrix<4, 1, Sacado::Fad::SLFad<double, 6>>, 2>
        cross_section_quaternion;
    std::array<Core::LinAlg::Matrix<4, 1, double>, 2> cross_section_quaternion_ref;

    //! Transformation matrices from beam interpolation.
    Core::LinAlg::Matrix<n_dof_beam_1 + n_dof_beam_2, 12> left_transformation_matrix{
        Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<12, n_dof_beam_1 + n_dof_beam_2> right_transformation_matrix{
        Core::LinAlg::Initialization::zero};
  };

  /**
   * \brief Data container for beam-to-beam coupling terms.
   */
  template <unsigned int n_dof_beam_1, unsigned int n_dof_beam_2>
  struct BeamToBeamCouplingTerms
  {
    //! Constraints
    Core::LinAlg::Matrix<3, 1> constraint{Core::LinAlg::Initialization::zero};

    //! Constraints linearized w.rt. beam kinematics.
    Core::LinAlg::Matrix<3, 12> constraint_lin_kinematic{Core::LinAlg::Initialization::zero};

    //! Residuum linearized w.r.t. Lagrange multipliers.
    Core::LinAlg::Matrix<12, 3> residuum_lin_lambda{Core::LinAlg::Initialization::zero};

    //! Evaluation data for direct stiffness contributions.
    std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 2> evaluation_data_rotation{};
  };

  /**
   * \brief Class for point-wise beam to beam mesh tying.
   */
  template <typename Beam1, unsigned int n_dof_beam_1, typename Beam2, unsigned int n_dof_beam_2>
  class BeamToBeamPointCouplingPair : public BeamContactPair
  {
   protected:
    //! FAD type for rotational coupling. The 6 dependent DOFs are the 3 rotational DOFs of each
    //! beam element.
    using scalar_type_rot = typename Sacado::Fad::SLFad<double, 6>;

    //! Total number of DOFs for both beams.
    static const unsigned int n_dof_total = n_dof_beam_1 + n_dof_beam_2;

    //! Array with DOFs per beam.
    static constexpr std::array<unsigned int, 2> n_dof_beam = {n_dof_beam_1, n_dof_beam_2};

   public:
    /**
     * \brief Initialize the pair.
     */
    BeamToBeamPointCouplingPair(BeamToBeamPointCouplingPairParameters parameters,
        const std::shared_ptr<GeometryPair::LineToLineEvaluationData>& line_to_line_evaluation_data)
        : BeamContactPair(),
          parameters_(parameters),
          line_to_line_evaluation_data_(line_to_line_evaluation_data) {};

    /**
     * \brief Setup the beam coupling pair.
     */
    void setup() override;

    /**
     * \brief Flag if this pair is assembled directly or not.
     */
    inline bool is_assembly_direct() const override
    {
      return parameters_.constraint_enforcement ==
             BeamToBeamPointCouplingPairParameters::ConstraintEnforcement::penalty_direct;
    };


    /**
     * \brief Things that need to be done in a separate loop before the actual evaluation loop over
     * all contact pairs. (derived)
     */
    void pre_evaluate() override {};

    /**
     * \brief Evaluate this contact element pair.
     */
    bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22) override
    {
      return false;
    }

    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     *
     * @param discret (in) Pointer to the discretization.
     * @param force_vector (in / out) Global force vector.
     * @param stiffness_matrix (in / out) Global stiffness matrix.
     * @param displacement_vector (in) Global displacement vector.
     */
    void evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

    /**
     * \brief No need to update pair state vectors, as everything is done in the
     * evaluate_and_assemble call.
     */
    void reset_state(const std::vector<double>& beam_centerline_dofvec,
        const std::vector<double>& solid_nodal_dofvec) override {};

    /**
     * \brief This pair is always active.
     */
    inline bool get_contact_flag() const override { return true; }

    /**
     * \brief Get number of active contact point pairs on this element pair. Not yet implemented.
     */
    unsigned int get_num_all_active_contact_point_pairs() const override
    {
      FOUR_C_THROW("get_num_all_active_contact_point_pairs not yet implemented!");
      return 0;
    };

    /**
     * \brief Get coordinates of all active contact points on element1. Not yet implemented.
     */
    void get_all_active_contact_point_coords_element1(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const override
    {
      FOUR_C_THROW("get_all_active_contact_point_coords_element1 not yet implemented!");
    }

    /**
     * \brief Get coordinates of all active contact points on element2. Not yet implemented.
     */
    void get_all_active_contact_point_coords_element2(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const override
    {
      FOUR_C_THROW("get_all_active_contact_point_coords_element2 not yet implemented!");
    }

    /**
     * \brief Get all (scalar) contact forces of this contact pair. Not yet implemented.
     */
    void get_all_active_beam_to_beam_visualization_values(std::vector<double>& forces,
        std::vector<double>& gaps, std::vector<double>& angles,
        std::vector<int>& types) const override
    {
      FOUR_C_THROW("get_all_active_contact_forces not yet implemented!");
    }

    /**
     * \brief Get energy of penalty contact. Not yet implemented.
     */
    double get_energy() const override
    {
      FOUR_C_THROW("get_energy not implemented yet!");
      return 0.0;
    }

    /**
     * \brief Print information about this beam contact element pair to screen.
     */
    void print(std::ostream& out) const override;

    /**
     * \brief Print this beam contact element pair to screen.
     */
    void print_summary_one_line_per_active_segment_pair(std::ostream& out) const override;

    /**
     * \brief Returns the type of this beam point coupling pair.
     */
    ContactPairType get_type() const override
    {
      return ContactPairType::beam_to_beam_point_coupling;
    }

    /**
     * \brief Evaluate the global matrices and vectors resulting from mortar coupling. (derived)
     */
    void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
        Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
        Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
        Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
        Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
        Core::LinAlg::FEVector<double>& global_lambda_active,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

    /**
     * \brief Evaluate the terms that directly assemble it into the global force vector and
     * stiffness matrix (derived).
     */
    void evaluate_and_assemble(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& global_lambda,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

   private:
    /**
     * \brief Evaluate the closest point projection for this pair.
     */
    void evaluate_closest_point_projection();

    /**
     * \brief Evaluate the kinematics for this pair.
     */
    [[nodiscard]] BeamToBeamKinematic<n_dof_beam_1, n_dof_beam_2> evaluate_kinematics(
        const Core::FE::Discretization& discret,
        const Core::LinAlg::Vector<double>& displacement_vector) const;

    /**
     * \brief Evaluate the positional coupling terms based on general cross-section kinematics.
     */
    [[nodiscard]] BeamToBeamCouplingTerms<n_dof_beam_1, n_dof_beam_2> evaluate_positional_coupling(
        const BeamToBeamKinematic<n_dof_beam_1, n_dof_beam_2>& pair_kinematic) const;

    /**
     * \brief Evaluate the rotational coupling terms based on general cross-section kinematics.
     */
    [[nodiscard]] BeamToBeamCouplingTerms<n_dof_beam_1, n_dof_beam_2> evaluate_rotational_coupling(
        const BeamToBeamKinematic<n_dof_beam_1, n_dof_beam_2>& pair_kinematic) const;

    /**
     * \brief Add the coupling stiffness and map the residuum and stiffness to the pair degrees of
     * freedom.
     */
    static void add_coupling_stiffness(Core::LinAlg::Matrix<12, 12>& stiffness,
        const BeamToBeamKinematic<n_dof_beam_1, n_dof_beam_2>& pair_kinematic,
        const BeamToBeamCouplingTerms<n_dof_beam_1, n_dof_beam_2>& coupling_terms_position,
        const Core::LinAlg::Matrix<3, 1>& lambda_position,
        const BeamToBeamCouplingTerms<n_dof_beam_1, n_dof_beam_2>& coupling_terms_rotation,
        const Core::LinAlg::Matrix<3, 1>& lambda_rotation);

    /**
     * \brief Map the residuum and stiffness to the pair degrees of freedom.
     */
    std::pair<Core::LinAlg::Matrix<n_dof_beam_1 + n_dof_beam_2, 1>,
        Core::LinAlg::Matrix<n_dof_beam_1 + n_dof_beam_2, n_dof_beam_1 + n_dof_beam_2>>
    map_residuum_and_stiffness_to_pair_dof(Core::LinAlg::Matrix<12, 12>& stiffness,
        const BeamToBeamKinematic<n_dof_beam_1, n_dof_beam_2>& pair_kinematic,
        const BeamToBeamCouplingTerms<n_dof_beam_1, n_dof_beam_2>& coupling_terms_position,
        const Core::LinAlg::Matrix<3, 1>& lambda_position,
        const BeamToBeamCouplingTerms<n_dof_beam_1, n_dof_beam_2>& coupling_terms_rotation,
        const Core::LinAlg::Matrix<3, 1>& lambda_rotation) const;

   private:
    //! Parameters for the coupling pair.
    BeamToBeamPointCouplingPairParameters parameters_;

    // Pointer to the two beam elements.
    std::array<const Core::Elements::Element*, 2> beam_elements_;

    //! Line to line evaluation data for closest point projection
    const std::shared_ptr<GeometryPair::LineToLineEvaluationData> line_to_line_evaluation_data_ =
        nullptr;

    //! Lagrange multiplier GIDs
    std::array<int, 6> lambda_gid_{-1};
  };  // namespace BeamInteraction

  /**
   * \brief Factory for beam-to-beam point coupling pairs.
   */
  std::unique_ptr<BeamInteraction::BeamContactPair> beam_to_beam_point_coupling_pair_factory(
      const std::array<const Core::Elements::Element*, 2>& element_ptrs,
      const BeamToBeamPointCouplingPairParameters& parameters,
      const std::shared_ptr<GeometryPair::LineToLineEvaluationData>& line_to_line_evaluation_data);

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
