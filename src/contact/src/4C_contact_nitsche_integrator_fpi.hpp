// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_INTEGRATOR_FPI_HPP
#define FOUR_C_CONTACT_NITSCHE_INTEGRATOR_FPI_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_integrator_poro.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace XFEM
{
  class XFluidContactComm;
}

namespace CONTACT
{
  class Element;

  class IntegratorNitscheFpi : public IntegratorNitschePoro
  {
   public:
    /*!
     \brief Constructor  with shape function specification

     Constructs an instance of this class using a specific type of shape functions.<br>
     Note that this is \b not a collective call as overlaps are
     integrated in parallel by individual processes.<br>
     Note also that this constructor relies heavily on the
     Core::FE::IntegrationPoints structs to get Gauss points
     and corresponding weights.

     */
    IntegratorNitscheFpi(Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm);
    //! @name Derived functions
    //! @{

    //! @name currently unsupported derived methods
    //! @{
    void integrate_deriv_segment_2d(Mortar::Element& source_elem, double& source_xi_a,
        double& source_xi_b, Mortar::Element& target_elem, double& target_xi_a, double& target_xi_b,
        MPI_Comm comm, const std::shared_ptr<Mortar::ParamsInterface>& cparams_ptr) override
    {
      FOUR_C_THROW("Segment based integration is currently unsupported!");
    }

    void integrate_deriv_ele_2d(Mortar::Element& source_elem,
        std::vector<Mortar::Element*> target_elems, bool* boundary_ele,
        const std::shared_ptr<Mortar::ParamsInterface>& cparams_ptr) override
    {
      FOUR_C_THROW("Element based integration in 2D is currently unsupported!");
    }

    void integrate_deriv_cell_3d_aux_plane(Mortar::Element& source_elem,
        Mortar::Element& target_elem, std::shared_ptr<Mortar::IntCell> cell, double* auxn,
        MPI_Comm comm, const std::shared_ptr<Mortar::ParamsInterface>& cparams_ptr) override
    {
      FOUR_C_THROW("The auxiliary plane 3-D coupling integration case is currently unsupported!");
    }
    //! @}

    /*!
     \brief First, reevaluate which gausspoints should be used
     Second, Build all integrals and linearizations without segmentation -- 3D
     (i.e. M, g, LinM, Ling and possibly D, LinD)
     */
    void integrate_deriv_ele_3d(Mortar::Element& source_elem,
        std::vector<Mortar::Element*> target_elems, bool* boundary_ele, bool* proj_, MPI_Comm comm,
        const std::shared_ptr<Mortar::ParamsInterface>& cparams_ptr) override;

    //! @}

   protected:
    /*!
     \brief Perform integration at GP
            This is where the distinction between methods should be,
            i.e. mortar, augmented, gpts,...
     */
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
     \brief Perform integration at GP
            This is where the distinction between methods should be,
            i.e. mortar, augmented, gpts,...
     */
    void integrate_gp_2d(Mortar::Element& source_elem, Mortar::Element& target_elem,
        Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
        Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseMatrix& source_deriv,
        Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
        double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
        Core::Gen::Pairedvector<int, double>& deriv_gap, double* source_xi, double* target_xi,
        std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
        std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi) override
    {
      FOUR_C_THROW("2d problems not available for IntegratorNitscheFsi, as CutFEM is only 3D!");
    }

   private:
    /*!
    \brief evaluate GPTS forces and linearization at this gp
    */
    template <int dim>
    void gpts_forces(Mortar::Element& source_elem, Mortar::Element& target_elem,
        const Core::LinAlg::SerialDenseVector& source_val,
        const Core::LinAlg::SerialDenseMatrix& source_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi,
        const Core::LinAlg::SerialDenseVector& target_val,
        const Core::LinAlg::SerialDenseMatrix& target_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi, const double jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
        const double gap, const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double* source_xi,
        double* target_xi);


    template <int dim>
    double get_normal_contact_transition(Mortar::Element& source_elem, Mortar::Element& target_elem,
        const Core::LinAlg::SerialDenseVector& source_val,
        const Core::LinAlg::SerialDenseVector& target_val, const double* source_xi,
        const Core::LinAlg::Matrix<dim, 1>& pxsi, const Core::LinAlg::Matrix<dim, 1>& normal,
        bool& FSI_integrated, bool& gp_on_this_proc);

    /// Update Element contact state -2...not_specified, -1...no_contact, 0...mixed, 1...contact
    void update_ele_contact_state(Mortar::Element& source_elem, int state);

    /// Element contact state -2...not_specified, -1...no_contact, 0...mixed, 1...contact
    int ele_contact_state_;

    /// Xfluid Contact Communicator
    std::shared_ptr<XFEM::XFluidContactComm> xf_c_comm_;
  };
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
