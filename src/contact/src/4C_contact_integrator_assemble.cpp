// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_node.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_coupling3d_classes.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Assemble g~ contribution (2D / 3D)                        popp 01/08|
 |  This method assembles the contribution of a 1D/2D source and target  |
 |  overlap pair to the weighted gap of the adjacent source nodes.       |
 *----------------------------------------------------------------------*/
bool CONTACT::Integrator::assemble_g(
    MPI_Comm comm, Mortar::Element& source_elem, Core::LinAlg::SerialDenseVector& gseg)
{
  // get adjacent source nodes to assemble to
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("AssembleG: Null pointer for source_nodes!");

  // loop over all source nodes
  for (int source = 0; source < source_elem.num_node(); ++source)
  {
    CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(source_nodes[source]);

    // only process source node rows that belong to this proc
    if (source_node->owner() != Core::Communication::my_mpi_rank(comm)) continue;

    // do not process source side boundary nodes
    // (their row entries would be zero anyway!)
    if (source_node->is_on_bound()) continue;

    double val = gseg(source);
    source_node->addg_value(val);
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  Assemble g~ contribution (2D / 3D)                        popp 02/10|
 |  PIECEWISE LINEAR LM INTERPOLATION VERSION                           |
 *----------------------------------------------------------------------*/
bool CONTACT::Integrator::assemble_g(
    MPI_Comm comm, Mortar::IntElement& sintele, Core::LinAlg::SerialDenseVector& gseg)
{
  // get adjacent source int nodes to assemble to
  Core::Nodes::Node** source_nodes = sintele.nodes();
  if (!source_nodes) FOUR_C_THROW("AssembleG: Null pointer for sintnodes!");

  // loop over all source nodes
  for (int source = 0; source < sintele.num_node(); ++source)
  {
    CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(source_nodes[source]);

    // only process source node rows that belong to this proc
    if (source_node->owner() != Core::Communication::my_mpi_rank(comm)) continue;

    // do not process source side boundary nodes
    // (their row entries would be zero anyway!)
    if (source_node->is_on_bound()) continue;

    double val = gseg(source);
    source_node->addg_value(val);
  }

  return true;
}

FOUR_C_NAMESPACE_CLOSE
