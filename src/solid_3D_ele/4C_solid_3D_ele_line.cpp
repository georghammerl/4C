// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele_line.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_solid_3D_ele_neumann_evaluator.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <string>

FOUR_C_NAMESPACE_OPEN


template <unsigned dim>
Discret::Elements::SolidLineType<dim> Discret::Elements::SolidLineType<dim>::instance_;

template <unsigned dim>
Discret::Elements::SolidLineType<dim>& Discret::Elements::SolidLineType<dim>::instance()
{
  return instance_;
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidLineType<dim>::create(
    const int id, const int owner)
{
  return nullptr;
}


template <unsigned dim>
Discret::Elements::SolidLine<dim>::SolidLine(const Discret::Elements::SolidLine<dim>& old) noexcept
    : Core::Elements::FaceElement(old), num_dof_per_node_(old.num_dof_per_node_)
{
}

template <unsigned dim>
Discret::Elements::SolidLine<dim>::SolidLine(int id, int owner, int nnode, const int* nodeids,
    Core::Nodes::Node** nodes, Core::Elements::Element* parent, const int lline)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_master_element(parent, lline);

  num_dof_per_node_ = parent_element()->num_dof_per_node(*SolidLine::nodes()[0]);
  // Safety check if all nodes have the same number of dofs!
  for (int nlid = 1; nlid < num_node(); ++nlid)
  {
    if (num_dof_per_node_ != parent_master_element()->num_dof_per_node(*SolidLine::nodes()[nlid]))
      FOUR_C_THROW("You need different NumDofPerNode for each node on this solid line? ({} != {})",
          num_dof_per_node_, parent_master_element()->num_dof_per_node(*SolidLine::nodes()[nlid]));
  }
}

template <unsigned dim>
Core::Elements::Element* Discret::Elements::SolidLine<dim>::clone() const
{
  auto* newelement = new Discret::Elements::SolidLine<dim>(*this);
  return newelement;
}

template <unsigned dim>
Core::FE::CellType Discret::Elements::SolidLine<dim>::shape() const
{
  return Core::FE::cell_type_switch(parent_element()->shape(),
      [&](auto celltype_t)
      {
        switch (num_node())
        {
          case 2:
            return Core::FE::is_nurbs<celltype_t()> ? Core::FE::CellType::nurbs2
                                                    : Core::FE::CellType::line2;
          case 3:
            return Core::FE::is_nurbs<celltype_t()> ? Core::FE::CellType::nurbs3
                                                    : Core::FE::CellType::line3;
          default:
            FOUR_C_THROW("unexpected number of nodes {}", num_node());
        }
      });
}

template <unsigned dim>
void Discret::Elements::SolidLine<dim>::pack(Core::Communication::PackBuffer& data) const
{
  data.add_to_pack(num_dof_per_node_);
}

template <unsigned dim>
void Discret::Elements::SolidLine<dim>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  buffer.extract_from_pack(num_dof_per_node_);
}

template <unsigned dim>
void Discret::Elements::SolidLine<dim>::print(std::ostream& os) const
{
  os << "SolidLine<" + std::to_string(dim) + "> ";
  Element::print(os);
}

template <unsigned dim>
int Discret::Elements::SolidLine<dim>::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);

  const double total_time = std::invoke(
      [&]()
      {
        if (parent_element()->is_params_interface())
          return parent_element()->params_interface_ptr()->get_total_time();
        else
          return params.get("total time", -1.0);
      });

  Discret::Elements::evaluate_neumann_by_element<dim>(
      *this, discretization, condition, elevec1, total_time);
  return 0;
}

template class Discret::Elements::SolidLine<3>;
template class Discret::Elements::SolidLine<2>;

FOUR_C_NAMESPACE_CLOSE
