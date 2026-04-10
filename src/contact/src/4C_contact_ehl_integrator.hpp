// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_EHL_INTEGRATOR_HPP
#define FOUR_C_CONTACT_EHL_INTEGRATOR_HPP

#include "4C_config.hpp"

#include "4C_contact_integrator.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_utils_pairedvector.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace CONTACT
{
  class IntegratorEhl : public CONTACT::Integrator
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
    IntegratorEhl(Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
        : Integrator(params, eletype, comm)
    {
    }


   protected:
    /*!
     \brief Perform integration at GP
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
        std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi) override;

    /*!
     \brief Perform integration at GP
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

   private:
    // integrate surface gradient
    void gp_weighted_surf_grad_and_deriv(Mortar::Element& source_elem, const double* xi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
        const Core::LinAlg::SerialDenseVector& lm_val,
        const Core::LinAlg::SerialDenseMatrix& lm_deriv,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        const Core::LinAlg::SerialDenseVector& source_val,
        const Core::LinAlg::SerialDenseMatrix& source_deriv,
        const Core::LinAlg::SerialDenseMatrix& sderiv2, const double& wgt, const double& jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap);

    // integrate relative and average tangential velocity
    void gp_weighted_av_rel_vel(Mortar::Element& source_elem, Mortar::Element& target_elem,
        const Core::LinAlg::SerialDenseVector& source_val,
        const Core::LinAlg::SerialDenseVector& lm_val,
        const Core::LinAlg::SerialDenseVector& target_val,
        const Core::LinAlg::SerialDenseMatrix& source_deriv,
        const Core::LinAlg::SerialDenseMatrix& target_deriv,
        const Core::LinAlg::SerialDenseMatrix& lm_deriv,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        const double& wgt, const double& jac, const Core::Gen::Pairedvector<int, double>& derivjac,
        const double* normal, const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit,
        const double& gap, const Core::Gen::Pairedvector<int, double>& deriv_gap,
        const double* source_xi, const double* target_xi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi);
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
