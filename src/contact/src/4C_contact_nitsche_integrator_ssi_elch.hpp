// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_INTEGRATOR_SSI_ELCH_HPP
#define FOUR_C_CONTACT_NITSCHE_INTEGRATOR_SSI_ELCH_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_integrator_ssi.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
   * @brief This class performs Gauss integration and the assembly to element matrices and vectors
   * that are relevant to the Nitsche contact formulation for scatra-structure interaction problems
   * using the electrochemistry formulation of the scatra field.
   *
   * @note Relevant methods are already templated w.r.t. the problem dimension. Currently only
   * 'dim=3' is used and tested but it should be quite easy to extend this if necessary.
   */
  class IntegratorNitscheSsiElch : public IntegratorNitscheSsi
  {
   public:
    /*!
     * @brief Constructor with shape function specification
     *
     * Constructs an instance of this class using a specific type of shape functions.<br> Note that
     * this is \b not a collective call as overlaps are integrated in parallel by individual
     * processes.<br> Note also that this constructor relies heavily on the
     * Core::FE::IntegrationPoints structs to get Gauss points and corresponding weights.
     *
     * @param[in] params   interface contact parameter list
     * @param[in] eletype  shape of integration cell for segment based integration or source side
     *                     mortar contact element for element based integration
     * @param[in] comm     contact interface communicator
     */
    IntegratorNitscheSsiElch(
        Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm);

   private:
    //! data bundle of current element
    template <int dim>
    struct ElementDataBundle;

    /*!
     * @brief Checks which element (target- or source-side) is the electrode-side and bundle the
     * data
     *
     * @tparam dim                    dimension of the problem
     * @param[in] source_ele           source side mortar element
     * @param[in] source_xi            source side coordinates in parameter space at current Gauss
     *                                point
     * @param[in] source_shape         source side shape function evaluated at current Gauss point
     * @param[in] source_shape_deriv   source side shape function derivative at current Gauss point
     * @param[in] source_normal        source side normal at current Gauss point
     * @param[in] d_source_xi_dd       directional derivative of source side Gauss point coordinates
     * @param[in] target_ele          target side mortar element
     * @param[in] target_xi           target side coordinates in parameter space at current Gauss
     *                                point
     * @param[in] target_shape        target side shape function evaluated at current Gauss point
     * @param[in] target_shape_deriv  target side shape function derivative at current Gauss point
     * @param[in] target_normal       target side normal at current Gauss point
     * @param[in] d_target_xi_dd      directional derivative of target side Gauss point coordinates
     * @param[out] source_is_electrode  flag indicating if source-side is electrode-side
     * @param[out] electrode_data      data bundle of the electrode-side element
     * @param[out] electrolyte_data    data bundle of the electrolyte-side element
     */
    template <int dim>
    void assign_electrode_and_electrolyte_quantities(Mortar::Element& source_ele, double* source_xi,
        const Core::LinAlg::SerialDenseVector& source_shape,
        const Core::LinAlg::SerialDenseMatrix& source_shape_deriv,
        const Core::LinAlg::Matrix<dim, 1>& source_normal,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_dd,
        Mortar::Element& target_ele, double* target_xi,
        const Core::LinAlg::SerialDenseVector& target_shape,
        const Core::LinAlg::SerialDenseMatrix& target_shape_deriv,
        const Core::LinAlg::Matrix<dim, 1>& target_normal,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_dd,
        bool& source_is_electrode, ElementDataBundle<dim>& electrode_quantities,
        ElementDataBundle<dim>& electrolyte_quantities);

    /*!
     * @brief calculate the determinant of the deformation gradient in the parent element at the
     * current Gauss point
     *
     * @tparam dim                      dimension of the problem
     * @param[in] electrode_quantities  data bundle of the electrode-side element
     * @return  determinant of the deformation gradient at the current Gauss point
     */
    template <int dim>
    double calculate_det_f_of_parent_element(const ElementDataBundle<dim>& electrode_quantities);

    /*!
     * @brief Calculates the derivative of the determinant of the deformation gradient w.r.t. the
     * displacement dofs
     *
     * @tparam dim  dimension of the problem
     * @param[in] detF                   determinant of the deformation gradient
     * @param[in] electrode_quantities   data bundle of the electrode-side element
     * @param[out] d_detF_dd             derivative of the determinant of the deformation gradient
     *                                   w.r.t. displacement
     */
    template <int dim>
    void calculate_spatial_derivative_of_det_f(double detF,
        const ElementDataBundle<dim>& electrode_quantities,
        Core::Gen::Pairedvector<int, double>& d_detF_dd);

    /*!
     * @brief Calculates the derivative of the determinant of the deformation gradient w.r.t. the
     * displacement dofs
     *
     * @tparam distype  shape of the element
     * @tparam dim      dimension of the problem
     * @param[in] detF                   determinant of the deformation gradient
     * @param[in] electrode_quantities   data bundle of the electrode-side element
     * @param[out] d_detF_dd             derivative of the determinant of the deformation gradient
     *                                   w.r.t. displacement
     */
    template <Core::FE::CellType distype, int dim>
    void calculate_spatial_derivative_of_det_f(double detF,
        const ElementDataBundle<dim>& electrode_quantities,
        Core::Gen::Pairedvector<int, double>& d_detF_dd);

    /*!
     * @brief evaluate gauss point to segment forces and linearization at this gp
     *
     * @tparam dim  dimension of the problem
     * @param[in] source_ele           source side mortar element
     * @param[in] target_ele          target side mortar element
     * @param[in] source_shape         source side shape function evaluated at current Gauss point
     * @param[in] source_shape_deriv   source side shape function derivative at current Gauss point
     * @param[in] d_source_xi_dd       directional derivative of source side Gauss point coordinates
     * @param[in] target_shape        target side shape function evaluated at current Gauss point
     * @param[in] target_shape_deriv  target side shape function derivative at current Gauss point
     * @param[in] d_target_xi_dd      directional derivative of target side Gauss point coordinates
     * @param[in] jac                 Jacobian determinant of integration cell
     * @param[in] d_jac_dd            directional derivative of cell Jacobian
     * @param[in] gp_wgt              Gauss point weight
     * @param[in] gap                 gap
     * @param[in] d_gap_dd            directional derivative of gap
     * @param[in] gp_normal           Gauss point normal
     * @param[in] d_gp_normal_dd      directional derivative of Gauss point normal
     * @param[in] source_xi            source side Gauss point coordinates
     * @param[in] target_xi           target side Gauss point coordinates
     */
    template <int dim>
    void gpts_forces(Mortar::Element& source_ele, Mortar::Element& target_ele,
        const Core::LinAlg::SerialDenseVector& source_shape,
        const Core::LinAlg::SerialDenseMatrix& source_shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_dd,
        const Core::LinAlg::SerialDenseVector& target_shape,
        const Core::LinAlg::SerialDenseMatrix& target_shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_dd, double jac,
        const Core::Gen::Pairedvector<int, double>& d_jac_dd, double gp_wgt, double gap,
        const Core::Gen::Pairedvector<int, double>& d_gap_dd, const double* gp_normal,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd, double* source_xi,
        double* target_xi);

    /*!
     * @brief integrate the electrochemistry residual and linearizations
     *
     * @tparam dim     dimension of the problem
     * @param[in] fac  pre-factor to correct sign dependent on integration of target or source side
     *                 terms
     * @param[in] ele_data_bundle  data bundle of current element
     * @param[in] jac              Jacobian determinant of integration cell
     * @param[in] d_jac_dd         directional derivative of cell Jacobian
     * @param[in] wgt              Gauss point weight
     * @param[in] test_val         quantity to be integrated
     * @param[in] d_test_val_dd    directional derivative of quantity to be integrated
     * @param[in] d_test_val_ds    derivative of quantity to be integrated w.r.t. scalar s
     */
    template <int dim>
    void integrate_elch_test(double fac, const ElementDataBundle<dim>& ele_data_bundle, double jac,
        const Core::Gen::Pairedvector<int, double>& d_jac_dd, double wgt, double test_val,
        const Core::Gen::Pairedvector<int, double>& d_test_val_dd,
        const Core::Gen::Pairedvector<int, double>& d_test_val_ds);

    void integrate_gp_3d(Mortar::Element& source_elem, Mortar::Element& target_elem,
        Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
        Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseMatrix& source_deriv,
        Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
        double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
        Core::Gen::Pairedvector<int, double>& deriv_gap, double* source_xi, double* target_xi,
        std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
        std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi) override;

    /*!
     * @brief integrate the scatra-structure interaction interface condition
     *
     * @tparam dim                    dimension of the problem
     * @param[in] source_is_electrode  flag indicating if source side is electrode side
     * @param[in] jac                 Jacobian determinant of integration cell
     * @param[in] d_jac_dd            directional derivative of cell Jacobian
     * @param[in] wgt                 Gauss point weight
     * @param[in] electrode_quantities    electrode element data bundle
     * @param[in] electrolyte_quantities  electrolyte element data bundle
     */
    template <int dim>
    void integrate_ssi_interface_condition(bool source_is_electrode, double jac,
        const Core::Gen::Pairedvector<int, double>& d_jac_dd, double wgt,
        const ElementDataBundle<dim>& electrode_quantities,
        const ElementDataBundle<dim>& electrolyte_quantities);

    /*!
     * @brief  integrate the structure residual and linearizations
     *
     * @tparam dim     dimension of the problem
     * @param[in] fac  pre-factor to correct sign dependent on integration of target or source side
     *                 terms
     * @param[in] ele  mortar contact element or integration cell mortar element
     * @param[in] shape          shape function evaluated at current Gauss point
     * @param[in] shape_deriv    shape function derivative at current Gauss point
     * @param[in] d_xi_dd        directional derivative of Gauss point coordinates
     * @param[in] jac            Jacobian determinant of integration cell
     * @param[in] d_jac_dd       directional derivative of cell Jacobian
     * @param[in] wgt            Gauss point weight
     * @param[in] test_val       quantity to be integrated
     * @param[in] d_test_val_dd  directional derivative of quantity to be integrated
     * @param[in] d_test_val_ds  derivative of quantity to be integrated w.r.t. scalar s
     * @param[in] normal         normal
     * @param[in] d_normal_dd    directional derivative of normal
     */
    template <int dim>
    void integrate_test(double fac, Mortar::Element& ele,
        const Core::LinAlg::SerialDenseVector& shape,
        const Core::LinAlg::SerialDenseMatrix& shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd, double jac,
        const Core::Gen::Pairedvector<int, double>& d_jac_dd, double wgt, double test_val,
        const Core::Gen::Pairedvector<int, double>& d_test_val_dd,
        const Core::Gen::Pairedvector<int, double>& d_test_val_ds,
        const Core::LinAlg::Matrix<dim, 1>& normal,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_normal_dd);

    /*!
     * @brief setup the electrochemistry Gauss point quantities
     *
     * @tparam dim                 dimension of the problem
     * @param[in] ele_data_bundle  data bundle of current element
     * @param[out] gp_conc         concentration at current Gauss point
     * @param[out] gp_pot          electric potential at current Gauss point
     * @param[out] d_conc_dc       derivative of Gauss point concentration w.r.t. concentration
     * @param[out] d_conc_dd       derivative of Gauss point concentration w.r.t. displacement
     * @param[out] d_pot_dpot      derivative of Gauss point electric potential w.r.t. electric
     *                             potential
     * @param[out] d_pot_dd        derivative of Gauss point electric potential w.r.t. displacement
     */
    template <int dim>
    void setup_gp_elch_properties(const ElementDataBundle<dim>& ele_data_bundle, double& gp_conc,
        double& gp_pot, Core::Gen::Pairedvector<int, double>& d_conc_dc,
        Core::Gen::Pairedvector<int, double>& d_conc_dd,
        Core::Gen::Pairedvector<int, double>& d_pot_dpot,
        Core::Gen::Pairedvector<int, double>& d_pot_dd);

    /*!
     * @brief  Evaluate cauchy stress component and its derivatives
     *
     * @tparam dim  dimension of the problem
     * @param[in] mortar_ele     mortar element
     * @param[in] gp_coord       Gauss point coordinates
     * @param[in] d_gp_coord_dd  directional derivative of Gauss point coordinates
     * @param[in] gp_wgt         Gauss point weight
     * @param[in] gp_normal      Gauss point normal
     * @param[in] d_gp_normal_dd directional derivative of Gauss point normal
     * @param[in] test_dir       direction of evaluation (e.g. normal or tangential direction)
     * @param[in] d_test_dir_dd  directional derivative of direction of evaluation
     * @param[in] nitsche_wgt    Nitsche weight
     * @param[out] cauchy_nt_wgt   Cauchy stress tensor contracted with normal vector n and
     *                             direction vector t multiplied by nitsche_wgt \f[ nitsche_wgt
     *                             \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot \boldsymbol{t} \f]
     * @param[out] d_cauchy_nt_dd  directional derivative of cauchy_nt \f[ \frac{ \mathrm{d}
     *                             \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot
     *                             \boldsymbol{t}}{\mathrm{d} \boldsymbol{d}} \f]
     * @param[out] d_cauchy_nt_de  derivative of cauchy_nt w.r.t. elch dofs e \f[ \frac{ \mathrm{d}
     *                             \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot
     *                             \boldsymbol{t}}{\mathrm{d} e} \f]
     */
    template <int dim>
    void so_ele_cauchy(Mortar::Element& mortar_ele, double* gp_coord,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_coord_dd, double gp_wgt,
        const Core::LinAlg::Matrix<dim, 1>& gp_normal,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd,
        const Core::LinAlg::Matrix<dim, 1>& test_dir,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_test_dir_dd, double nitsche_wgt,
        double& cauchy_nt_wgt, Core::Gen::Pairedvector<int, double>& d_cauchy_nt_dd,
        Core::Gen::Pairedvector<int, double>& d_cauchy_nt_de);

    //! number of dofs per node
    static constexpr int numdofpernode_ = 2;
  };
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
