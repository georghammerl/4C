// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_coupling2d.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_integrator_factory.hpp"
#include "4C_contact_interpolator.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 06/09|
 *----------------------------------------------------------------------*/
CONTACT::Coupling2d::Coupling2d(Core::FE::Discretization& idiscret, int dim, bool quad,
    Teuchos::ParameterList& params, Mortar::Element& source_elem, Mortar::Element& target_elem)
    : Mortar::Coupling2d(idiscret, dim, quad, params, source_elem, target_elem),
      stype_(Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params, "STRATEGY"))
{
  // empty constructor

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate source / target overlap (public)                 popp 04/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling2d::integrate_overlap(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("IntegrateOverlap called without specific shape function defined!");

  /**********************************************************************/
  /* INTEGRATION                                                        */
  /* Depending on overlap and the xiproj_ entries integrate the Mortar  */
  /* matrices D and M and the weighted gap function g~ on the overlap   */
  /* of the current sl / ma pair.                                       */
  /**********************************************************************/

  // no integration if no overlap
  if (!overlap_) return false;

  // set segmentation status of all source nodes
  // (hassegment_ of a source node is true if ANY segment/cell
  // is integrated that contributes to this source node)
  int nnodes = source_element().num_node();
  Core::Nodes::Node** mynodes = source_element().nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");
  for (int k = 0; k < nnodes; ++k)
  {
    Mortar::Node* mycnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mycnode) FOUR_C_THROW("Null pointer!");
    mycnode->has_segment() = true;
  }

  // local working copies of input variables
  double source_xi_a = xiproj_[0];
  double source_xi_b = xiproj_[1];
  double target_xi_a = xiproj_[2];
  double target_xi_b = xiproj_[3];

  // create a CONTACT integrator instance with correct num_gp and Dim
  std::shared_ptr<CONTACT::Integrator> integrator =
      CONTACT::INTEGRATOR::build_integrator(stype_, imortar_, source_element().shape(), get_comm());
  // *******************************************************************
  // different options for mortar integration
  // *******************************************************************
  // (1) no quadratic element(s) involved -> linear LM interpolation
  // (2) quadratic element(s) involved -> quadratic LM interpolation
  // (3) quadratic element(s) involved -> linear LM interpolation
  // (4) quadratic element(s) involved -> piecew. linear LM interpolation
  // *******************************************************************
  Mortar::LagMultQuad lmtype = lag_mult_quad();

  // *******************************************************************
  // cases (1), (2) and (3)
  // *******************************************************************
  if (!quad() || (quad() && lmtype == Mortar::lagmult_quad) ||
      (quad() && lmtype == Mortar::lagmult_lin) || (quad() && lmtype == Mortar::lagmult_const))
  {
    // ***********************************************************
    //                   Integrate stuff !!!                    //
    // ***********************************************************
    integrator->integrate_deriv_segment_2d(source_element(), source_xi_a, source_xi_b,
        target_element(), target_xi_a, target_xi_b, get_comm(), mparams_ptr);
    // ***********************************************************
    //                   END INTEGRATION !!!                    //
    // ***********************************************************
  }

  // *******************************************************************
  // case (4)
  // *******************************************************************
  else if (quad() && lmtype == Mortar::lagmult_pwlin)
  {
    FOUR_C_THROW("Piecewise linear LM not (yet?) implemented in 2D");
  }

  // *******************************************************************
  // undefined case
  // *******************************************************************
  else if (quad() && lmtype == Mortar::lagmult_undefined)
  {
    FOUR_C_THROW(
        "Lagrange multiplier interpolation for quadratic elements undefined\n"
        "If you are using 2nd order mortar elements, you need to specify LM_QUAD in MORTAR "
        "COUPLING section");
  }

  // *******************************************************************
  // other cases
  // *******************************************************************
  else
  {
    FOUR_C_THROW("Invalid case for 2D mortar coupling LM interpolation");
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 06/09|
 *----------------------------------------------------------------------*/
CONTACT::Coupling2dManager::Coupling2dManager(Core::FE::Discretization& idiscret, int dim,
    bool quad, Teuchos::ParameterList& params, Mortar::Element* source_elem,
    std::vector<Mortar::Element*> target_elem)
    : Mortar::Coupling2dManager(idiscret, dim, quad, params, source_elem, target_elem),
      stype_(Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params, "STRATEGY"))
{
  // empty constructor
  return;
}


/*----------------------------------------------------------------------*
 |  get communicator  (public)                               farah 01/13|
 *----------------------------------------------------------------------*/
MPI_Comm CONTACT::Coupling2dManager::get_comm() const { return idiscret_.get_comm(); }

/*----------------------------------------------------------------------*
 |  Evaluate coupling pairs                                  farah 10/14|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling2dManager::evaluate_coupling(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  if (target_elements().size() == 0) return false;

  // decide which type of coupling should be evaluated
  auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  //*********************************
  // Mortar Contact
  //*********************************
  if (algo == Mortar::algorithm_mortar || algo == Mortar::algorithm_gpts)
    integrate_coupling(mparams_ptr);

  //*********************************
  // Error
  //*********************************
  else
    FOUR_C_THROW("chose contact algorithm not supported!");

  return true;
}


/*----------------------------------------------------------------------*
 |  Evaluate mortar coupling pairs                           Popp 03/09 |
 *----------------------------------------------------------------------*/
void CONTACT::Coupling2dManager::integrate_coupling(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  //**********************************************************************
  // STANDARD INTEGRATION (SEGMENTS)
  //**********************************************************************
  if (int_type() == Mortar::inttype_segments)
  {
    // loop over all target elements associated with this source element
    for (int it = 0; it < (int)target_elements().size(); ++it)
    {
      // create Coupling2d object and push back
      coupling().push_back(std::make_shared<Coupling2d>(
          idiscret_, dim_, quad_, imortar_, source_element(), target_element(it)));

      // project the element pair
      coupling()[it]->project();

      // check for element overlap
      coupling()[it]->detect_overlap();
    }

    // calculate consistent dual shape functions for this element
    consistent_dual_shape();

    // do mortar integration
    for (int it = 0; it < (int)target_elements().size(); ++it)
      coupling()[it]->integrate_overlap(mparams_ptr);

    // free memory of consistent dual shape function coefficient matrix
    source_element().mo_data().reset_dual_shape();
    source_element().mo_data().reset_deriv_dual_shape();
  }
  //**********************************************************************
  // FAST INTEGRATION (ELEMENTS)
  //**********************************************************************
  else if (int_type() == Mortar::inttype_elements or int_type() == Mortar::inttype_elements_BS)
  {
    if ((int)target_elements().size() == 0) return;

    // create an integrator instance with correct num_gp and Dim
    std::shared_ptr<CONTACT::Integrator> integrator = CONTACT::INTEGRATOR::build_integrator(
        stype_, imortar_, source_element().shape(), get_comm());

    // *******************************************************************
    // different options for mortar integration
    // *******************************************************************
    // (1) no quadratic element(s) involved -> linear LM interpolation
    // (2) quadratic element(s) involved -> quadratic LM interpolation
    // (3) quadratic element(s) involved -> linear LM interpolation
    // (4) quadratic element(s) involved -> piecew. linear LM interpolation
    // *******************************************************************
    Mortar::LagMultQuad lmtype = lag_mult_quad();

    // *******************************************************************
    // cases (1), (2) and (3)
    // *******************************************************************
    if (!quad() || (quad() && lmtype == Mortar::lagmult_quad) ||
        (quad() && lmtype == Mortar::lagmult_lin))
    {
      // Test whether projection from source to target surface is feasible -->
      // important for dual LM Fnc.
      // Contact_interface.cpp --> AssembleG
      for (unsigned it = 0; it < target_elements().size(); ++it)
      {
        // create Coupling2d object and push back
        coupling().push_back(std::make_shared<Coupling2d>(
            idiscret_, dim_, quad_, imortar_, source_element(), target_element(it)));

        // project the element pair
        coupling()[it]->project();
      }

      // Bool for identification of boundary elements
      bool boundary_ele = false;

      // ***********************************************************
      //                  START INTEGRATION !!!                   //
      // ***********************************************************
      integrator->integrate_deriv_ele_2d(
          source_element(), target_elements(), &boundary_ele, mparams_ptr);
      // ***********************************************************
      //                   END INTEGRATION !!!                    //
      // ***********************************************************

      if (int_type() == Mortar::inttype_elements_BS and boundary_ele == true)
      {
        // switch, if consistent boundary modification chosen
        if (Teuchos::getIntegralValue<Mortar::ConsistentDualType>(imortar_,
                "LM_DUAL_CONSISTENT") == Mortar::ConsistentDualType::consistent_boundary &&
            shape_fcn() != Mortar::shape_standard  // so for petrov-Galerkin and dual
        )
        {
          // loop over all target elements associated with this source element
          for (int it = 0; it < (int)target_elements().size(); ++it)
          {
            // create Coupling2d object and push back
            coupling().push_back(std::make_shared<Coupling2d>(
                idiscret_, dim_, quad_, imortar_, source_element(), target_element(it)));

            // project the element pair
            coupling()[it]->project();

            // check for element overlap
            coupling()[it]->detect_overlap();
          }

          // calculate consistent dual shape functions for this element
          consistent_dual_shape();

          // do mortar integration
          for (int it = 0; it < (int)target_elements().size(); ++it)
            coupling()[it]->integrate_overlap(mparams_ptr);

          // free memory of consistent dual shape function coefficient matrix
          source_element().mo_data().reset_dual_shape();
          source_element().mo_data().reset_deriv_dual_shape();
        }

        // segment-based integration for boundary elements
        else
        {
          // loop over all target elements associated with this source element
          for (int it = 0; it < (int)target_elements().size(); ++it)
          {
            // create Coupling2d object and push back
            coupling().push_back(std::make_shared<Coupling2d>(
                idiscret_, dim_, quad_, imortar_, source_element(), target_element(it)));

            // project the element pair
            coupling()[it]->project();

            // check for element overlap
            coupling()[it]->detect_overlap();

            // integrate the element overlap
            coupling()[it]->integrate_overlap(mparams_ptr);
          }
        }
      }
      else
      {
        // nothing to do...
      }
    }
    // *******************************************************************
    // case (4)
    // *******************************************************************
    else if (quad() && lmtype == Mortar::lagmult_pwlin)
    {
      FOUR_C_THROW("Piecewise linear LM not (yet?) implemented in 2D");
    }

    // *******************************************************************
    // undefined case
    // *******************************************************************
    else if (quad() && lmtype == Mortar::lagmult_undefined)
    {
      FOUR_C_THROW(
          "Lagrange multiplier interpolation for quadratic elements undefined\n"
          "If you are using 2nd order mortar elements, you need to specify LM_QUAD"
          " in MORTAR COUPLING section");
    }

    // *******************************************************************
    // other cases
    // *******************************************************************
    else
    {
      FOUR_C_THROW("Integrate: Invalid case for 2D mortar coupling LM interpolation");
    }
  }
  //**********************************************************************
  // INVALID TYPE OF NUMERICAL INTEGRATION
  //**********************************************************************
  else
  {
    FOUR_C_THROW("Invalid type of numerical integration");
  }
}


/*----------------------------------------------------------------------*
 |  Calculate dual shape functions                           seitz 07/13|
 *----------------------------------------------------------------------*/
void CONTACT::Coupling2dManager::consistent_dual_shape()
{
  // For standard shape functions no modification is necessary
  // A switch earlier in the process improves computational efficiency
  auto consistent =
      Teuchos::getIntegralValue<Mortar::ConsistentDualType>(imortar_, "LM_DUAL_CONSISTENT");
  if (shape_fcn() == Mortar::shape_standard || consistent == Mortar::consistent_none) return;

  // Consistent modification not yet checked for constant LM interpolation
  if (quad() == true && lag_mult_quad() == Mortar::lagmult_const &&
      consistent != Mortar::consistent_none)
    FOUR_C_THROW("Consistent dual shape functions not yet checked for constant LM interpolation!");

  // do nothing if there are no coupling pairs
  if (coupling().size() == 0) return;

  const int nnodes = source_element().num_node();
  const int ndof = 2;

  int linsize = 0;
  for (int i = 0; i < nnodes; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(source_element().nodes()[i]);
    linsize += cnode->get_linsize();
  }

  int target_nodes = 0;
  for (int it = 0; it < (int)coupling().size(); ++it)
    target_nodes += target_elements()[it]->num_node();

  // detect entire overlap
  double ximin = 1.0;
  double ximax = -1.0;
  Core::Gen::Pairedvector<int, double> dximin(linsize + ndof * target_nodes);
  Core::Gen::Pairedvector<int, double> dximax(linsize + ndof * target_nodes);

  // loop over all target elements associated with this source element
  for (int it = 0; it < (int)coupling().size(); ++it)
  {
    double source_xi_a = coupling()[it]->xi_proj()[0];
    double source_xi_b = coupling()[it]->xi_proj()[1];
    double target_xi_a = coupling()[it]->xi_proj()[2];
    double target_xi_b = coupling()[it]->xi_proj()[3];

    // no overlap for this source-target pair --> continue with next pair
    if (source_xi_a == 0.0 && source_xi_b == 0.0) continue;

    // for contact we need the derivatives as well
    bool startsource = false;
    bool endsource = false;

    if (source_xi_a == -1.0)
      startsource = true;
    else
      startsource = false;
    if (source_xi_b == 1.0)
      endsource = true;
    else
      endsource = false;

    // create an integrator for this segment
    CONTACT::Integrator integrator(imortar_, source_element().shape(), get_comm());

    std::vector<Core::Gen::Pairedvector<int, double>> ximaps(4, linsize + ndof * target_nodes);
    // get directional derivatives of source_xi_a, source_xi_b, target_xi_a, target_xi_b
    integrator.deriv_xi_a_b_2d(source_element(), source_xi_a, source_xi_b, target_element(it),
        target_xi_a, target_xi_b, ximaps, startsource, endsource, linsize);

    // get element contact integration area
    // and for contact derivatives of beginning and end
    if ((source_xi_a != 0.0 || source_xi_b != 0.0) && (source_xi_a >= -1.0 && source_xi_a <= 1.0) &&
        (source_xi_b >= -1.0 && source_xi_b <= 1.0))
    {
      if (source_xi_a < ximin && source_xi_a >= -1. && source_xi_a <= 1.)
      {
        ximin = source_xi_a;
        dximin = ximaps[0];
      }
      if (source_xi_b > ximax && source_xi_b >= -1. && source_xi_b <= 1.)
      {
        ximax = source_xi_b;
        dximax = ximaps[1];
      }
    }
  }

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // no overlap: the applied dual shape functions don't matter, as the integration domain is void
  if ((ximax == -1.0 && ximin == 1.0) || (ximax - ximin < 4. * MORTARINTLIM)) return;

  // fully projecting element: no modification necessary
  if (ximin == -1.0 && ximax == 1.0) return;

  // calculate consistent dual schape functions (see e.g. Cichosz et.al.:
  // Consistent treatment of boundaries with mortar contact formulations, CMAME 2010

  // store derivae into element
  source_element().mo_data().deriv_dual_shape() =
      std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
          linsize + 2 * ndof * target_nodes, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivae =
      *(source_element().mo_data().deriv_dual_shape());

  // compute entries to bi-ortho matrices me/de with Gauss quadrature
  Mortar::ElementIntegrator integrator(source_element().shape());

  // prepare for calculation of dual shape functions
  Core::LinAlg::SerialDenseMatrix me(nnodes, nnodes, true);
  Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);
  // two-dim arrays of maps for linearization of me/de
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivme(nnodes,
      std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, linsize + 2 * ndof * target_nodes));
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivde(nnodes,
      std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, linsize + 2 * ndof * target_nodes));

  Core::LinAlg::SerialDenseVector source_val(nnodes);
  Core::LinAlg::SerialDenseMatrix source_deriv(nnodes, 1, true);
  Core::LinAlg::SerialDenseMatrix ssecderiv(nnodes, 1);

  for (int gp = 0; gp < integrator.n_gp(); ++gp)
  {
    // coordinates and weight
    std::array<double, 2> eta = {integrator.coordinate(gp, 0), 0.0};
    double wgt = integrator.weight(gp);

    // coordinate transformation source_xi->eta (source Mortar::Element->Overlap)
    double source_xi[2] = {0.0, 0.0};
    source_xi[0] = 0.5 * (1.0 - eta[0]) * ximin + 0.5 * (1.0 + eta[0]) * ximax;

    // evaluate trace space shape functions
    if (lag_mult_quad() == Mortar::lagmult_lin)
      source_element().evaluate_shape_lag_mult_lin(
          Mortar::shape_standard, source_xi, source_val, source_deriv, nnodes);
    else
      source_element().evaluate_shape(source_xi, source_val, source_deriv, nnodes);
    source_element().evaluate2nd_deriv_shape(source_xi, ssecderiv, nnodes);

    // evaluate the two source side Jacobians
    double dxdsxi = source_element().jacobian(source_xi);
    double dsxideta = -0.5 * ximin + 0.5 * ximax;

    // evaluate linearizations *******************************************
    // evaluate the derivative dxdsxidsxi = Jac,xi
    double djacdxi[2] = {0.0, 0.0};
    dynamic_cast<CONTACT::Element&>(source_element()).d_jac_d_xi(djacdxi, source_xi, ssecderiv);
    double dxdsxidsxi = djacdxi[0];  // only 2D here

    // evaluate the GP source coordinate derivatives
    Core::Gen::Pairedvector<int, double> d_source_xi_gp(linsize + ndof * target_nodes);
    for (CI p = dximin.begin(); p != dximin.end(); ++p)
      d_source_xi_gp[p->first] += 0.5 * (1 - eta[0]) * (p->second);
    for (CI p = dximax.begin(); p != dximax.end(); ++p)
      d_source_xi_gp[p->first] += 0.5 * (1 + eta[0]) * (p->second);

    // evaluate the Jacobian derivative
    Core::Gen::Pairedvector<int, double> derivjac(source_element().num_node() * n_dim());
    source_element().deriv_jacobian(source_xi, derivjac);

    // integrate dual shape matrices de, me and their linearizations
    for (int j = 0; j < nnodes; ++j)
    {
      double fac;
      // de and linearization
      de(j, j) += wgt * source_val[j] * dxdsxi * dsxideta;

      // (1) linearization of source gp coordinates in ansatz function j for derivative of de
      fac = wgt * source_deriv(j, 0) * dxdsxi * dsxideta;
      for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
        derivde[j][j][p->first] += fac * (p->second);

      // (2) linearization dsxideta - segment end coordinates
      fac = 0.5 * wgt * source_val[j] * dxdsxi;
      for (CI p = dximin.begin(); p != dximin.end(); ++p)
        derivde[j][j][p->first] -= fac * (p->second);
      fac = 0.5 * wgt * source_val[j] * dxdsxi;
      for (CI p = dximax.begin(); p != dximax.end(); ++p)
        derivde[j][j][p->first] += fac * (p->second);

      // (3) linearization dxdsxi - source GP jacobian
      fac = wgt * source_val[j] * dsxideta;
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        derivde[j][j][p->first] += fac * (p->second);

      // (4) linearization dxdsxi - source GP coordinates
      fac = wgt * source_val[j] * dsxideta * dxdsxidsxi;
      for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
        derivde[j][j][p->first] += fac * (p->second);

      // me and linearization
      for (int k = 0; k < nnodes; ++k)
      {
        me(j, k) += wgt * source_val[j] * source_val[k] * dxdsxi * dsxideta;

        // (1) linearization of source gp coordinates in ansatz function for derivative of me
        fac = wgt * source_val[k] * dxdsxi * dsxideta * source_deriv(j, 0);
        for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
        {
          derivme[j][k][p->first] += fac * (p->second);
          derivme[k][j][p->first] += fac * (p->second);
        }

        // (2) linearization dsxideta - segment end coordinates
        fac = 0.5 * wgt * source_val[j] * source_val[k] * dxdsxi;
        for (CI p = dximin.begin(); p != dximin.end(); ++p)
          derivme[j][k][p->first] -= fac * (p->second);
        fac = 0.5 * wgt * source_val[j] * source_val[k] * dxdsxi;
        for (CI p = dximax.begin(); p != dximax.end(); ++p)
          derivme[j][k][p->first] += fac * (p->second);

        // (3) linearization dxdsxi - source GP jacobian
        fac = wgt * source_val[j] * source_val[k] * dsxideta;
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          derivme[j][k][p->first] += fac * (p->second);

        // (4) linearization dxdsxi - source GP coordinates
        fac = wgt * source_val[j] * source_val[k] * dsxideta * dxdsxidsxi;
        for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
          derivme[j][k][p->first] += fac * (p->second);
      }
    }
  }

  // declare dual shape functions coefficient matrix and
  // inverse of matrix M_e
  Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);
  Core::LinAlg::SerialDenseMatrix meinv(nnodes, nnodes, true);

  // compute matrix A_e and inverse of matrix M_e for
  // linear interpolation of quadratic element
  if (lag_mult_quad() == Mortar::lagmult_lin)
  {
    // how many non-zero nodes
    const int nnodeslin = 2;

    // reduce me to non-zero nodes before inverting
    Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin;
    for (int j = 0; j < nnodeslin; ++j)
      for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

    // invert bi-ortho matrix melin
    Core::LinAlg::inverse(melin);

    // re-inflate inverse of melin to full size
    for (int j = 0; j < nnodeslin; ++j)
      for (int k = 0; k < nnodeslin; ++k) meinv(j, k) = melin(j, k);

    // get solution matrix with dual parameters
    Core::LinAlg::multiply(ae, de, meinv);
  }
  // compute matrix A_e and inverse of matrix M_e for all other cases
  else
    meinv = Core::LinAlg::invert_and_multiply_by_cholesky(me, de, ae);

  // build linearization of ae and store in derivdual
  // (this is done according to a quite complex formula, which
  // we get from the linearization of the biorthogonality condition:
  // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )

  // loop over all entries of ae (index i,j)
  for (int i = 0; i < nnodes; ++i)
  {
    for (int j = 0; j < nnodes; ++j)
    {
      // compute Lin(Ae) according to formula above
      for (int l = 0; l < nnodes; ++l)  // loop over sum l
      {
        // part1: Lin(De)*Inv(Me)
        for (CI p = derivde[i][l].begin(); p != derivde[i][l].end(); ++p)
          derivae[p->first](i, j) += meinv(l, j) * (p->second);

        // part2: Ae*Lin(Me)*Inv(Me)
        for (int k = 0; k < nnodes; ++k)  // loop over sum k
          for (CI p = derivme[k][l].begin(); p != derivme[k][l].end(); ++p)
            derivae[p->first](i, j) -= ae(i, k) * meinv(l, j) * (p->second);
      }
    }
  }

  // store ae matrix in source element data container
  source_element().mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);

  return;
}

FOUR_C_NAMESPACE_CLOSE
